"""
Telco Customer Churn - Baseline Modeling with MLflow Tracking
=============================================================
Trains multiple baseline models (Logistic Regression, Random Forest,
Gradient Boosting) and logs everything to MLflow.

Artifacts logged:
- Model parameters
- Metrics (accuracy, precision, recall, F1, AUC-ROC)
- Model artifact (sklearn model)
- confusion_matrix.png (custom artifact 1)
- classification_report.json (custom artifact 2)

Author: Eko Muhammad Rizki
Project: SMSML_Eko_Muhammad_Rizki
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
# Try connecting to MLflow tracking server, fallback to local tracking if unavailable
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
try:
    import requests
    requests.get(MLFLOW_TRACKING_URI, timeout=2)
except Exception:
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_NAME = "telco-churn-baseline"
DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"


# ============================================================
# Utility Functions
# ============================================================
def load_processed_data(data_dir: str = DATA_DIR) -> tuple:
    """Load preprocessed train/test data."""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    print(f"[INFO] Loaded data — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str) -> str:
    """Generate and save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix: {path}")
    return path


def save_classification_report(y_true, y_pred, model_name: str, save_dir: str) -> str:
    """Generate and save classification report as JSON."""
    report = classification_report(
        y_true, y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
    )
    report["model_name"] = model_name
    report["timestamp"] = datetime.now().isoformat()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "classification_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved classification report: {path}")
    return path


# ============================================================
# Model Definitions
# ============================================================
def get_baseline_models() -> dict:
    """Return dictionary of baseline models with their parameters."""
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
                C=1.0,
            ),
            "params": {
                "max_iter": 1000,
                "random_state": 42,
                "solver": "lbfgs",
                "C": 1.0,
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": 42,
            },
        },
    }


# ============================================================
# Training & Logging
# ============================================================
def train_and_log_model(
    model_name: str,
    model_config: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train a single model and log everything to MLflow.

    Logs:
    - All model parameters
    - Metrics: accuracy, precision, recall, f1, auc_roc
    - Model artifact via mlflow.sklearn.log_model
    - Custom artifact 1: confusion_matrix.png
    - Custom artifact 2: classification_report.json
    """
    model = model_config["model"]
    params = model_config["params"]

    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_type", model_name)

        # Train
        start_time = datetime.now()
        model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        mlflow.log_metric("training_time_seconds", train_time)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
        }

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"  Model artifact logged successfully")

        # Custom Artifact 1: Confusion Matrix PNG
        artifact_dir = os.path.join(ARTIFACTS_DIR, model_name)
        cm_path = save_confusion_matrix(y_test, y_pred, model_name, artifact_dir)
        mlflow.log_artifact(cm_path, artifact_path="custom_artifacts")

        # Custom Artifact 2: Classification Report JSON
        cr_path = save_classification_report(y_test, y_pred, model_name, artifact_dir)
        mlflow.log_artifact(cr_path, artifact_path="custom_artifacts")

        # Log run info
        run_id = mlflow.active_run().info.run_id
        print(f"  MLflow Run ID: {run_id}")
        print(f"  Training time: {train_time:.2f}s")

        return {
            "model_name": model_name,
            "run_id": run_id,
            "metrics": metrics,
            "training_time": train_time,
        }


# ============================================================
# Main Execution
# ============================================================
def main(data_dir: str = DATA_DIR):
    """Run baseline model training with MLflow tracking."""
    print("=" * 60)
    print("TELCO CHURN — BASELINE MODELING WITH MLFLOW")
    print("=" * 60)

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[INFO] MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"[INFO] Experiment: {EXPERIMENT_NAME}")

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    # Train all baseline models
    models = get_baseline_models()
    results = []

    for model_name, model_config in models.items():
        result = train_and_log_model(
            model_name, model_config, X_train, X_test, y_train, y_test
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Accuracy':<10} {'F1':<10} {'AUC-ROC':<10}")
    print("-" * 55)
    for r in results:
        m = r["metrics"]
        print(f"{r['model_name']:<25} {m['accuracy']:<10.4f} {m['f1_score']:<10.4f} {m['auc_roc']:<10.4f}")

    # Find best model
    best = max(results, key=lambda x: x["metrics"]["f1_score"])
    print(f"\n[SUCCESS] Best model: {best['model_name']} (F1: {best['metrics']['f1_score']:.4f})")
    print(f"   MLflow Run ID: {best['run_id']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Modeling with MLflow")
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100,
        help="Number of estimators for tree models",
    )
    parser.add_argument(
        "--max_depth", type=int, default=10,
        help="Max depth for tree models",
    )
    args = parser.parse_args()

    main(data_dir=args.data_dir)
