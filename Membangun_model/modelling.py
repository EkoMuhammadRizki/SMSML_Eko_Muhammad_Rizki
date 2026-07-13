"""
Telco Customer Churn - Baseline Modeling with MLflow Tracking
=============================================================
Trains multiple baseline models (Logistic Regression, Random Forest,
Gradient Boosting) and logs everything to MLflow.

Artifacts logged via autolog (log_models=True):
- Model parameters (automatic)
- Training metrics (automatic)
- Model artifact folder: MLmodel, conda.yaml, model.pkl,
  python_env.yaml, requirements.txt, estimator.html

Custom artifacts logged manually:
- confusion_matrix.png
- classification_report.json
- training_history.png (comparison across models)

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
    roc_curve,
)
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "telco-churn-baseline"
DATA_DIR = "preprocessing/namadataset_preprocessing"
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


def save_training_history(results: list, save_dir: str) -> str:
    """
    Generate and save training_history.png — a grouped bar chart
    comparing key metrics across all trained models.
    """
    model_names = [r["model_name"] for r in results]
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    x = np.arange(len(model_names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]

    for i, metric in enumerate(metrics_to_plot):
        values = [r["metrics"][metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(),
                      color=colors[i], edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Training History — Model Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved training history: {path}")
    return path


def save_roc_curves(results_with_proba: list, save_dir: str) -> str:
    """Generate and save ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for (model_name, y_true, y_proba), color in zip(results_with_proba, colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{model_name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    plt.title("ROC Curves — All Models", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ROC curves: {path}")
    return path


# ============================================================
# Model Definitions
# ============================================================
def get_baseline_models(n_estimators: int = 100, max_depth: int = 10) -> dict:
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
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
            "params": {
                "n_estimators": n_estimators,
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
    Train a single model and log to MLflow.

    mlflow.autolog(log_models=True) handles:
    - All model parameters
    - Training metrics
    - Model artifact folder (MLmodel, conda.yaml, model.pkl,
      python_env.yaml, requirements.txt, estimator.html)

    Custom artifacts logged manually:
    - confusion_matrix.png
    - classification_report.json
    """
    model = model_config["model"]

    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        # Train — autolog automatically logs params, metrics, and model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate and log additional metrics manually
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "training_time_seconds": train_time,
        }

        # Log metrics that autolog may not capture
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("auc_roc", metrics["auc_roc"])
        mlflow.log_metric("training_time_seconds", train_time)

        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # NOTE: model artifact is logged automatically by autolog(log_models=True)
        # No need for manual mlflow.sklearn.log_model() call
        print(f"  Model artifact logged automatically by autolog")

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
            "y_test": y_test,
            "y_pred_proba": y_pred_proba,
        }


# ============================================================
# Main Execution
# ============================================================
def main(data_dir: str = DATA_DIR, n_estimators: int = 100, max_depth: int = 10):
    """Run baseline model training with MLflow tracking and autolog."""
    print("=" * 60)
    print("TELCO CHURN — BASELINE MODELING WITH MLFLOW")
    print("=" * 60)

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable MLflow autolog with log_models=True
    # This ensures the full model/ folder is created:
    #   MLmodel, conda.yaml, model.pkl, python_env.yaml,
    #   requirements.txt, estimator.html
    mlflow.sklearn.autolog(log_models=True)
    print(f"[INFO] MLflow sklearn.autolog enabled (log_models=True)")
    print(f"[INFO] MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"[INFO] Experiment: {EXPERIMENT_NAME}")

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    # Train all baseline models
    models = get_baseline_models(n_estimators=n_estimators, max_depth=max_depth)
    results = []
    roc_data = []

    for model_name, model_config in models.items():
        result = train_and_log_model(
            model_name, model_config, X_train, X_test, y_train, y_test
        )
        results.append(result)
        roc_data.append((model_name, result["y_test"], result["y_pred_proba"]))

    # Generate and log training_history.png as a summary artifact
    history_path = save_training_history(results, ARTIFACTS_DIR)
    roc_path = save_roc_curves(roc_data, ARTIFACTS_DIR)

    # Log training_history and ROC curves in a separate summary run
    with mlflow.start_run(run_name="Training_Summary"):
        mlflow.log_artifact(history_path, artifact_path="summary")
        mlflow.log_artifact(roc_path, artifact_path="summary")

        # Log best model info as params
        best = max(results, key=lambda x: x["metrics"]["f1_score"])
        mlflow.log_param("best_model", best["model_name"])
        mlflow.log_metric("best_f1_score", best["metrics"]["f1_score"])
        mlflow.log_metric("best_auc_roc", best["metrics"]["auc_roc"])
        print(f"\n[INFO] Training history and ROC curves logged as summary artifacts")

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

    main(data_dir=args.data_dir, n_estimators=args.n_estimators, max_depth=args.max_depth)
