"""
Telco Customer Churn - Prometheus Metrics Exporter
====================================================
Custom Prometheus metrics exporter for ML inference monitoring.
Exposes 12 custom metrics covering inference, system, and data drift indicators.

Metrics exposed:
 1. ml_inference_latency_seconds     (Histogram) — Inference request latency
 2. ml_request_total                 (Counter)   — Total prediction requests
 3. ml_http_errors_total             (Counter)   — HTTP error responses (4xx/5xx)
 4. ml_prediction_churn_total        (Counter)   — Predictions classified as churn
 5. ml_prediction_no_churn_total     (Counter)   — Predictions classified as no churn
 6. ml_prediction_confidence         (Histogram) — Model confidence score distribution
 7. ml_data_drift_score              (Gauge)     — Feature drift indicator score
 8. ml_active_connections            (Gauge)     — Current active connections
 9. system_cpu_usage_percent         (Gauge)     — CPU utilization
10. system_memory_usage_percent      (Gauge)     — Memory utilization
11. system_disk_usage_percent        (Gauge)     — Disk utilization
12. ml_model_reload_total            (Counter)   — Model hot-reload count

Author: Eko Muhammad Rizki
Project: SMSML_Eko_Muhammad_Rizki
"""

import os
import time
import threading
import logging
from typing import Optional

import numpy as np
import psutil
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)
from fastapi import FastAPI
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# ============================================================
# Custom Prometheus Metrics
# ============================================================

# 1. Inference latency (Histogram)
ML_INFERENCE_LATENCY = Histogram(
    "ml_inference_latency_seconds",
    "Inference request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# 2. Total request count (Counter)
ML_REQUEST_TOTAL = Counter(
    "ml_request_total",
    "Total prediction requests",
    ["method", "endpoint", "status"],
)

# 3. HTTP errors (Counter)
ML_HTTP_ERRORS = Counter(
    "ml_http_errors_total",
    "Total HTTP error responses (4xx/5xx)",
    ["status_code", "endpoint"],
)

# 4. Churn predictions (Counter)
ML_PREDICTION_CHURN = Counter(
    "ml_prediction_churn_total",
    "Total predictions classified as churn",
)

# 5. No-churn predictions (Counter)
ML_PREDICTION_NO_CHURN = Counter(
    "ml_prediction_no_churn_total",
    "Total predictions classified as no churn",
)

# 6. Prediction confidence (Histogram)
ML_PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Model confidence score distribution",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)

# 7. Data drift score (Gauge)
ML_DATA_DRIFT_SCORE = Gauge(
    "ml_data_drift_score",
    "Feature drift indicator score (0=no drift, 1=max drift)",
)

# 8. Active connections (Gauge)
ML_ACTIVE_CONNECTIONS = Gauge(
    "ml_active_connections",
    "Current active connections to the inference server",
)

# 9. CPU usage (Gauge)
SYSTEM_CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "Current CPU utilization percentage",
)

# 10. Memory usage (Gauge)
SYSTEM_MEMORY_USAGE = Gauge(
    "system_memory_usage_percent",
    "Current memory utilization percentage",
)

# 11. Disk usage (Gauge)
SYSTEM_DISK_USAGE = Gauge(
    "system_disk_usage_percent",
    "Current disk utilization percentage",
)

# 12. Model reload count (Counter)
ML_MODEL_RELOAD = Counter(
    "ml_model_reload_total",
    "Total number of model hot-reload events",
)


# ============================================================
# Metrics Collector Class
# ============================================================
class MetricsCollector:
    """
    Collects and updates all Prometheus metrics.
    Designed to be used as a singleton in the FastAPI application.
    """

    def __init__(self):
        self._active_connections = 0
        self._lock = threading.Lock()

        # Start background system metrics updater
        self._start_system_metrics_thread()
        logger.info("MetricsCollector initialized with 12 custom metrics")

    def track_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        latency: float,
    ):
        """Track an HTTP request's metrics."""
        # 1. Latency
        ML_INFERENCE_LATENCY.observe(latency)

        # 2. Request count
        ML_REQUEST_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()

        # 3. HTTP errors
        if status_code >= 400:
            ML_HTTP_ERRORS.labels(
                status_code=str(status_code),
                endpoint=endpoint,
            ).inc()

    def track_prediction(
        self,
        prediction: int,
        confidence: float,
        inference_time: float,
    ):
        """Track a model prediction's metrics."""
        # 4 & 5. Prediction counts
        if prediction == 1:
            ML_PREDICTION_CHURN.inc()
        else:
            ML_PREDICTION_NO_CHURN.inc()

        # 6. Confidence distribution
        ML_PREDICTION_CONFIDENCE.observe(confidence)

    def check_data_drift(self, new_data, reference_data):
        """
        Calculate a simple data drift score based on feature distribution shifts.
        Uses the Population Stability Index (PSI) approximation.
        """
        try:
            drift_scores = []
            numerical_cols = reference_data.select_dtypes(include=[np.number]).columns

            for col in numerical_cols:
                if col in new_data.columns:
                    ref_mean = reference_data[col].mean()
                    ref_std = reference_data[col].std()
                    if ref_std > 0:
                        new_val = new_data[col].values[0]
                        # Z-score based drift indicator
                        z_score = abs((new_val - ref_mean) / ref_std)
                        drift_scores.append(min(z_score / 3, 1.0))  # Normalize to [0, 1]

            if drift_scores:
                avg_drift = float(np.mean(drift_scores))
                ML_DATA_DRIFT_SCORE.set(avg_drift)
        except Exception as e:
            logger.warning(f"Drift detection error: {e}")

    def track_connection_open(self):
        """Track when a new connection is opened."""
        with self._lock:
            self._active_connections += 1
            ML_ACTIVE_CONNECTIONS.set(self._active_connections)

    def track_connection_close(self):
        """Track when a connection is closed."""
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)
            ML_ACTIVE_CONNECTIONS.set(self._active_connections)

    def track_model_reload(self):
        """Track a model hot-reload event."""
        ML_MODEL_RELOAD.inc()

    def track_error(self):
        """Track a generic error (increments error counter)."""
        ML_HTTP_ERRORS.labels(status_code="500", endpoint="/predict").inc()

    def _update_system_metrics(self):
        """Update system resource metrics (CPU, memory, disk)."""
        try:
            # 9. CPU
            cpu_pct = psutil.cpu_percent(interval=None)
            SYSTEM_CPU_USAGE.set(cpu_pct)

            # 10. Memory
            mem = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(mem.percent)

            # 11. Disk
            disk = psutil.disk_usage("/")
            SYSTEM_DISK_USAGE.set(disk.percent)

        except Exception as e:
            logger.warning(f"System metrics update error: {e}")

    def _start_system_metrics_thread(self):
        """Start a background thread that updates system metrics every 15 seconds."""
        def _updater():
            while True:
                self._update_system_metrics()
                time.sleep(15)

        thread = threading.Thread(target=_updater, daemon=True, name="system-metrics")
        thread.start()
        logger.info("System metrics background thread started (15s interval)")


# ============================================================
# FastAPI Metrics Endpoint
# ============================================================
def setup_metrics_endpoint(app: FastAPI):
    """
    Add the /metrics endpoint to a FastAPI application.
    This endpoint is scraped by Prometheus.
    """

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    logger.info("Prometheus /metrics endpoint registered")


# ============================================================
# Standalone Mode (for testing)
# ============================================================
if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="Prometheus Exporter Test")
    setup_metrics_endpoint(app)

    collector = MetricsCollector()

    @app.get("/test")
    async def test_endpoint():
        collector.track_request("GET", "/test", 200, 0.05)
        collector.track_prediction(prediction=1, confidence=0.87, inference_time=0.05)
        return {"status": "ok", "message": "Metrics updated"}

    print("Starting Prometheus exporter test server on :8000")
    print("Visit http://localhost:8000/metrics to see metrics")
    uvicorn.run(app, host="0.0.0.0", port=8000)
