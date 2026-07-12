"""
Telco Customer Churn - FastAPI Inference Server
=================================================
Production-ready inference endpoint with Prometheus metrics integration.
Serves the best MLflow model for real-time churn predictions.

Endpoints:
- POST /predict        — Single prediction
- POST /predict/batch  — Batch predictions
- GET  /health         — Health check
- GET  /metrics        — Prometheus metrics (via prometheus_exporter)
- GET  /model-info     — Model metadata

Author: Eko Muhammad Rizki
Project: SMSML_Eko_Muhammad_Rizki
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# Import Prometheus metrics from exporter (using spec_from_file_location since file starts with a number and has dots)
import importlib.util
dir_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(dir_path, "3.prometheus_exporter.py")

spec = importlib.util.spec_from_file_location("prometheus_exporter_module", file_path)
prometheus_exporter = importlib.util.module_from_spec(spec)
sys.modules["prometheus_exporter_module"] = prometheus_exporter
spec.loader.exec_module(prometheus_exporter)

MetricsCollector = prometheus_exporter.MetricsCollector
setup_metrics_endpoint = prometheus_exporter.setup_metrics_endpoint

# ============================================================
# Configuration
# ============================================================
MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/tuning/model")
MLFLOW_MODEL_URI = os.environ.get("MLFLOW_MODEL_URI", "models:/telco-churn-rf-tuned/latest")
FEATURE_NAMES_PATH = os.environ.get("FEATURE_NAMES_PATH", "data/processed/feature_names.json")
SCALER_PATH = os.environ.get("SCALER_PATH", "data/processed/scaler.joblib")
REFERENCE_DATA_PATH = os.environ.get("REFERENCE_DATA_PATH", "data/processed/X_train.csv")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Pydantic Schemas
# ============================================================
class CustomerData(BaseModel):
    """Input schema for a single customer prediction."""
    gender: int = Field(..., description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., description="0=No, 1=Yes")
    Partner: int = Field(..., description="0=No, 1=Yes")
    Dependents: int = Field(..., description="0=No, 1=Yes")
    tenure: float = Field(..., description="Number of months")
    PhoneService: int = Field(..., description="0=No, 1=Yes")
    PaperlessBilling: int = Field(..., description="0=No, 1=Yes")
    MonthlyCharges: float = Field(..., description="Monthly charge amount")
    TotalCharges: float = Field(..., description="Total charges")
    charge_per_month_ratio: float = Field(0.0, description="TotalCharges / (tenure+1)")
    is_new_customer: int = Field(0, description="1 if tenure <= 6")
    has_internet_addon: int = Field(0, description="Count of internet add-on services")
    # One-hot encoded features (defaults to 0)
    MultipleLines_No_phone_service: Optional[int] = 0
    MultipleLines_Yes: Optional[int] = 0
    InternetService_Fiber_optic: Optional[int] = 0
    InternetService_No: Optional[int] = 0
    OnlineSecurity_No_internet_service: Optional[int] = 0
    OnlineSecurity_Yes: Optional[int] = 0
    OnlineBackup_No_internet_service: Optional[int] = 0
    OnlineBackup_Yes: Optional[int] = 0
    DeviceProtection_No_internet_service: Optional[int] = 0
    DeviceProtection_Yes: Optional[int] = 0
    TechSupport_No_internet_service: Optional[int] = 0
    TechSupport_Yes: Optional[int] = 0
    StreamingTV_No_internet_service: Optional[int] = 0
    StreamingTV_Yes: Optional[int] = 0
    StreamingMovies_No_internet_service: Optional[int] = 0
    StreamingMovies_Yes: Optional[int] = 0
    Contract_One_year: Optional[int] = 0
    Contract_Two_year: Optional[int] = 0
    PaymentMethod_Credit_card_automatic: Optional[int] = 0
    PaymentMethod_Electronic_check: Optional[int] = 0
    PaymentMethod_Mailed_check: Optional[int] = 0

    class Config:
        json_schema_extra = {
            "example": {
                "gender": 1,
                "SeniorCitizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "tenure": 24.0,
                "PhoneService": 1,
                "PaperlessBilling": 1,
                "MonthlyCharges": 70.35,
                "TotalCharges": 1688.40,
                "charge_per_month_ratio": 67.54,
                "is_new_customer": 0,
                "has_internet_addon": 3,
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    prediction: int = Field(..., description="0=No Churn, 1=Churn")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability_churn: float = Field(..., description="Probability of churn")
    probability_no_churn: float = Field(..., description="Probability of no churn")
    confidence: float = Field(..., description="Model confidence (max probability)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""
    customers: list[CustomerData]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    timestamp: str


# ============================================================
# Application State
# ============================================================
class AppState:
    """Global application state."""
    model = None
    feature_names = None
    scaler = None
    reference_data = None
    metrics_collector = None
    start_time = None
    model_loaded = False
    model_reload_count = 0


state = AppState()


# ============================================================
# Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic."""
    # Startup
    state.start_time = datetime.now()
    logger.info("Starting Telco Churn Inference Server...")

    # Load model
    try:
        # Try loading from MLflow model registry first
        try:
            state.model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
            logger.info(f"Loaded model from MLflow registry: {MLFLOW_MODEL_URI}")
        except Exception:
            # Fallback: load from local artifacts
            model_path = os.path.join(MODEL_DIR, "model.pkl")
            if os.path.exists(model_path):
                state.model = joblib.load(model_path)
                logger.info(f"Loaded model from local path: {model_path}")
            else:
                # Try any sklearn model in mlruns
                import glob
                pattern = "mlruns/**/model/model.pkl"
                found = glob.glob(pattern, recursive=True)
                if found:
                    state.model = joblib.load(found[0])
                    logger.info(f"Loaded model from mlruns: {found[0]}")
                else:
                    logger.warning("No model found! Server will start but predictions will fail.")

        state.model_loaded = state.model is not None
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        state.model_loaded = False

    # Load feature names
    try:
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, "r") as f:
                state.feature_names = json.load(f)
            logger.info(f"Loaded {len(state.feature_names)} feature names")
    except Exception as e:
        logger.warning(f"Could not load feature names: {e}")

    # Load reference data for drift detection
    try:
        if os.path.exists(REFERENCE_DATA_PATH):
            state.reference_data = pd.read_csv(REFERENCE_DATA_PATH)
            logger.info("Loaded reference data for drift detection")
    except Exception as e:
        logger.warning(f"Could not load reference data: {e}")

    # Initialize metrics collector
    state.metrics_collector = MetricsCollector()
    logger.info("Prometheus metrics collector initialized")

    logger.info("[SUCCESS] Inference server ready!")

    yield

    # Shutdown
    logger.info("Shutting down inference server...")


# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML inference API for predicting customer churn with Prometheus monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Setup Prometheus metrics endpoint
setup_metrics_endpoint(app)


# ============================================================
# Middleware: Request Tracking
# ============================================================
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics via Prometheus."""
    start = time.time()

    try:
        response = await call_next(request)
        latency = time.time() - start

        if state.metrics_collector:
            state.metrics_collector.track_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                latency=latency,
            )

        return response
    except Exception as exc:
        latency = time.time() - start
        if state.metrics_collector:
            state.metrics_collector.track_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=500,
                latency=latency,
            )
        raise exc


# ============================================================
# Endpoints
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - state.start_time).total_seconds() if state.start_time else 0
    return HealthResponse(
        status="healthy" if state.model_loaded else "degraded",
        model_loaded=state.model_loaded,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model-info")
async def model_info():
    """Return model metadata."""
    return {
        "model_type": type(state.model).__name__ if state.model else "Not loaded",
        "model_loaded": state.model_loaded,
        "feature_count": len(state.feature_names) if state.feature_names else 0,
        "feature_names": state.feature_names,
        "reload_count": state.model_reload_count,
        "uptime_seconds": (datetime.now() - state.start_time).total_seconds()
        if state.start_time else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """
    Predict churn for a single customer.

    Returns prediction, probabilities, confidence, and inference latency.
    """
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    try:
        # Convert input to DataFrame
        input_data = customer.model_dump()
        df = pd.DataFrame([input_data])

        # Align columns with trained feature set
        if state.feature_names:
            for col in state.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[state.feature_names]

        # Predict
        prediction = int(state.model.predict(df)[0])
        probabilities = state.model.predict_proba(df)[0]

        inference_time = (time.time() - start) * 1000  # ms

        # Track prediction metrics
        if state.metrics_collector:
            state.metrics_collector.track_prediction(
                prediction=prediction,
                confidence=float(max(probabilities)),
                inference_time=time.time() - start,
            )

            # Data drift detection
            if state.reference_data is not None:
                state.metrics_collector.check_data_drift(
                    df, state.reference_data
                )

        return PredictionResponse(
            prediction=prediction,
            prediction_label="Churn" if prediction == 1 else "No Churn",
            probability_churn=round(float(probabilities[1]), 4),
            probability_no_churn=round(float(probabilities[0]), 4),
            confidence=round(float(max(probabilities)), 4),
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        if state.metrics_collector:
            state.metrics_collector.track_error()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers in a single request.
    """
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    results = []

    for customer in request.customers:
        input_data = customer.model_dump()
        df = pd.DataFrame([input_data])

        if state.feature_names:
            for col in state.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[state.feature_names]

        prediction = int(state.model.predict(df)[0])
        probabilities = state.model.predict_proba(df)[0]

        results.append({
            "prediction": prediction,
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "probability_churn": round(float(probabilities[1]), 4),
            "probability_no_churn": round(float(probabilities[0]), 4),
            "confidence": round(float(max(probabilities)), 4),
        })

    batch_time = (time.time() - start) * 1000

    return {
        "predictions": results,
        "total_samples": len(results),
        "batch_inference_time_ms": round(batch_time, 2),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Run Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
