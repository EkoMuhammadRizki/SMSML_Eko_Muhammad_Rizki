"""
3.prometheus_exporter.py
--------------------------
Exposes a /metrics endpoint (Prometheus format) that tracks the health and
performance of the locally-served MLflow model. Acts as a thin proxy: it
forwards prediction requests to the real `mlflow models serve` endpoint
while recording latency, request counts, error counts, and prediction
confidence as Prometheus metrics.

Run AFTER starting the served model:
    mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --env-manager=local

Then start the exporter:
    python "3.prometheus_exporter.py"

Exporter will be available at:
    http://localhost:8000/metrics       (Prometheus scrape target)
    http://localhost:8000/invocations   (proxy endpoint used by clients / Inference.py)
"""

import time
import logging

import requests
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_SERVING_URL = "http://127.0.0.1:5001/invocations"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "model_prediction_requests_total",
    "Total number of prediction requests received",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Latency of prediction requests in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)

PREDICTION_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Confidence score (max softmax probability) of predictions",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

MODEL_UP = Gauge(
    "model_serving_up",
    "Whether the underlying MLflow model serving endpoint is reachable (1=up, 0=down)",
)

PREDICTIONS_BY_CLASS = Counter(
    "model_predictions_by_class_total",
    "Total predictions grouped by predicted class",
    ["class_name"],
)

CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]


@app.route("/invocations", methods=["POST"])
def proxy_invocations():
    start_time = time.time()
    try:
        resp = requests.post(
            MODEL_SERVING_URL,
            headers={"Content-Type": "application/json"},
            data=request.data,
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()

        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(status="success").inc()
        MODEL_UP.set(1)

        predictions = result.get("predictions", result)
        if predictions:
            import numpy as np
            probs = np.array(predictions[0])
            confidence = float(np.max(probs))
            pred_class = CLASS_NAMES[int(np.argmax(probs))]
            PREDICTION_CONFIDENCE.observe(confidence)
            PREDICTIONS_BY_CLASS.labels(class_name=pred_class).inc()

        return jsonify(result)

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        MODEL_UP.set(0)
        logger.error(f"Inference proxy error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    try:
        requests.get(MODEL_SERVING_URL.replace("/invocations", "/ping"), timeout=3)
        MODEL_UP.set(1)
        return jsonify({"status": "ok"}), 200
    except Exception:
        MODEL_UP.set(0)
        return jsonify({"status": "unreachable"}), 503


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    logger.info("Starting Prometheus exporter on port 8000 ...")
    app.run(host="0.0.0.0", port=8000)
