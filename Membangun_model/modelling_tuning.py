"""
Telco Customer Churn - Hyperparameter Tuning with MLflow & DagsHub
===================================================================
Performs hyperparameter tuning using GridSearchCV/RandomizedSearchCV
on Random Forest with integrated MLflow tracking and DagsHub remote backend.

Artifacts logged:
- All tuning parameters & best parameters
- Metrics (accuracy, precision, recall, F1, AUC-ROC)
- Best model artifact
- confusion_matrix.png (custom artifact 1)
- classification_report.json (custom artifact 2)

Author: Eko Muhammad Rizki
Project: SMSML_Eko_Muhammad_Rizki
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
import dagshub

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts/tuning"
EXPERIMENT_NAME = "telco-churn-tuning"
RANDOM_STATE = 42

# DagsHub + MLflow Remote Tracking
DAGSHUB_OWNER = "EkoMuhammadRizki"
DAGSHUB_REPO = "SMSML_Eko_Muhammad_Rizki"


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
        cmap="Purples",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title(f"Confusion Matrix — {model_name} (Tuned)", fontsize=14, fontweight="bold")
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
    report["tuning_method"] = "GridSearchCV"
    report["timestamp"] = datetime.now().isoformat()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "classification_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved classification report: {path}")
    return path


# ============================================================
# Hyperparameter Search Space
# ============================================================
def get_param_grid() -> dict:
    """Return hyperparameter search grid for Random Forest."""
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }


# ============================================================
# Tuning Pipeline
# ============================================================
def run_tuning(data_dir: str = DATA_DIR):
    """
    Run hyperparameter tuning with GridSearchCV and log to MLflow + DagsHub.

    Uses nested MLflow runs:
    - Parent run: overall tuning session
    - Child runs: each combination explored by GridSearchCV
    """
    print("=" * 60)
    print("TELCO CHURN — HYPERPARAMETER TUNING WITH MLFLOW + DAGSHUB")
    print("=" * 60)

    # ── DagsHub Remote Tracking ──────────────────────────────
    dagshub.init(
        repo_owner=DAGSHUB_OWNER,
        repo_name=DAGSHUB_REPO,
        mlflow=True,
    )
    print(f"[INFO] DagsHub initialized: {DAGSHUB_OWNER}/{DAGSHUB_REPO}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[INFO] MLflow Experiment: {EXPERIMENT_NAME}")

    # ── Load Data ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    # ── Hyperparameter Grid ──────────────────────────────────
    param_grid = get_param_grid()
    print(f"[INFO] Param grid: {param_grid}")

    # ── Parent MLflow Run ────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest_Tuning_Session"):

        # Log tuning configuration
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("model_type", "RandomForest")

        # GridSearchCV
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        print("\n[INFO] Starting GridSearchCV...")
        start_time = datetime.now()
        grid_search.fit(X_train, y_train)
        tuning_time = (datetime.now() - start_time).total_seconds()

        print(f"[INFO] GridSearchCV completed in {tuning_time:.2f}s")
        print(f"[INFO] Best CV F1 Score: {grid_search.best_score_:.4f}")
        print(f"[INFO] Best Parameters: {grid_search.best_params_}")

        # Log tuning time and best CV score
        mlflow.log_metric("tuning_time_seconds", tuning_time)
        mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)

        # Log best parameters
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        # ── Evaluate Best Model on Test Set ──────────────────
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
        }

        # Log metrics
        print(f"\n{'='*60}")
        print("BEST MODEL — TEST SET METRICS")
        print(f"{'='*60}")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")

        # ── Log Model Artifact ───────────────────────────────
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="telco-churn-rf-tuned",
        )
        print(f"  Model artifact logged and registered")

        # ── Custom Artifact 1: Confusion Matrix ──────────────
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        cm_path = save_confusion_matrix(
            y_test, y_pred, "RandomForest_Tuned", ARTIFACTS_DIR
        )
        mlflow.log_artifact(cm_path, artifact_path="custom_artifacts")

        # ── Custom Artifact 2: Classification Report ─────────
        cr_path = save_classification_report(
            y_test, y_pred, "RandomForest_Tuned", ARTIFACTS_DIR
        )
        mlflow.log_artifact(cr_path, artifact_path="custom_artifacts")

        # ── Log Top-N GridSearch Results ─────────────────────
        cv_results = pd.DataFrame(grid_search.cv_results_)
        top_n = cv_results.nsmallest(10, "rank_test_score")[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ]
        top_n_path = os.path.join(ARTIFACTS_DIR, "top_cv_results.json")
        top_n.to_json(top_n_path, orient="records", indent=2)
        mlflow.log_artifact(top_n_path, artifact_path="custom_artifacts")
        print(f"  Top CV results logged")

        # ── Nested Runs for Top 5 Combinations ──────────────
        print(f"\n[INFO] Logging top 5 combinations as nested runs...")
        top_5 = cv_results.nsmallest(5, "rank_test_score")

        for idx, row in top_5.iterrows():
            with mlflow.start_run(
                run_name=f"Combo_Rank_{row['rank_test_score']}",
                nested=True,
            ):
                # Log parameters
                for param_key, param_val in row["params"].items():
                    mlflow.log_param(param_key, param_val)

                # Log CV metrics
                mlflow.log_metric("mean_cv_f1", row["mean_test_score"])
                mlflow.log_metric("std_cv_f1", row["std_test_score"])
                mlflow.log_metric("rank", row["rank_test_score"])

        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Tuning completed!")
        print(f"   Best F1: {metrics['f1_score']:.4f}")
        print(f"   Best AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"   MLflow Run ID: {run_id}")
        print(f"   DagsHub: https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}")


if __name__ == "__main__":
    run_tuning()
