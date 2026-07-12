"""
Telco Customer Churn - Data Preprocessing Pipeline
===================================================
Modular preprocessing script for the Telco Customer Churn dataset.
Converts raw data into ML-ready features with full pipeline automation.

Author: Eko Muhammad Rizki
Project: SMSML_Eko_Muhammad_Rizki
"""

import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]

MULTI_CATEGORY_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
# 1. Data Loading
# ============================================================
def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset from a local path or URL.

    Args:
        data_path: Path to the CSV file. If None or not found, downloads from URL.

    Returns:
        Raw DataFrame.
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from local path: {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info(f"Downloading data from URL: {DATASET_URL}")
        df = pd.read_csv(DATASET_URL)

    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df


# ============================================================
# 2. Handle Missing Values
# ============================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    - TotalCharges has empty strings that need to be converted to NaN then filled.
    - Drop customerID as it's not a feature.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()

    # Drop customerID (not a useful feature)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped 'customerID' column")

    # TotalCharges: convert empty strings to NaN, then to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        missing_count = df["TotalCharges"].isna().sum()
        if missing_count > 0:
            # Fill with median (robust to outliers)
            median_val = df["TotalCharges"].median()
            df["TotalCharges"].fillna(median_val, inplace=True)
            logger.info(
                f"Filled {missing_count} missing TotalCharges values with median: {median_val:.2f}"
            )

    # Check for any remaining missing values
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Remaining missing values: {remaining_missing}")
        df = df.dropna()
        logger.info("Dropped rows with remaining missing values")
    else:
        logger.info("No remaining missing values")

    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicate rows")
    else:
        logger.info("No duplicate rows found")

    logger.info(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================
# 3. Feature Engineering
# ============================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features to enhance model performance.

    New features:
    - tenure_group: Categorized tenure into groups
    - charge_per_month_ratio: TotalCharges / (tenure + 1)
    - is_new_customer: tenure <= 6 months
    - has_internet_addon: count of internet add-on services
    - monthly_charge_category: binned MonthlyCharges

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()

    # Tenure groups (0-12, 13-24, 25-48, 49-60, 61+)
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ["0-12", "13-24", "25-48", "49-60", "61+"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    logger.info("Created 'tenure_group' feature")

    # Charge per month ratio
    df["charge_per_month_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)
    logger.info("Created 'charge_per_month_ratio' feature")

    # Is new customer (tenure <= 6 months)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    logger.info("Created 'is_new_customer' feature")

    # Count of internet add-on services
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["has_internet_addon"] = df[addon_cols].apply(
        lambda row: sum(1 for val in row if val == "Yes"), axis=1
    )
    logger.info("Created 'has_internet_addon' feature")

    # Monthly charge category
    charge_bins = [0, 30, 60, 90, np.inf]
    charge_labels = ["Low", "Medium", "High", "Very High"]
    df["monthly_charge_category"] = pd.cut(
        df["MonthlyCharges"], bins=charge_bins, labels=charge_labels
    )
    logger.info("Created 'monthly_charge_category' feature")

    logger.info(f"After feature engineering: {df.shape[1]} columns")
    return df


# ============================================================
# 4. Encode Features
# ============================================================
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features.
    - Binary columns: LabelEncoder (0/1)
    - Multi-category columns: One-Hot Encoding (pd.get_dummies)

    Args:
        df: DataFrame with engineered features.

    Returns:
        Tuple of (encoded DataFrame, encoders dict for later inverse transform).
    """
    df = df.copy()
    encoders = {}

    # Binary encoding with LabelEncoder
    for col in BINARY_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"Label encoded '{col}': {list(le.classes_)}")

    # One-Hot Encoding for multi-category columns
    ohe_cols = [col for col in MULTI_CATEGORY_COLS if col in df.columns]
    # Also encode engineered categorical features
    engineered_cats = ["tenure_group", "monthly_charge_category"]
    ohe_cols += [col for col in engineered_cats if col in df.columns]

    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)
        logger.info(f"One-Hot Encoded columns: {ohe_cols}")

    logger.info(f"After encoding: {df.shape[1]} columns")
    return df, encoders


# ============================================================
# 5. Scale Features
# ============================================================
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: list[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    Fit on training data, transform both train and test.

    Args:
        X_train: Training features.
        X_test: Test features.
        numerical_cols: Columns to scale. Defaults to NUMERICAL_COLS.

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler).
    """
    if numerical_cols is None:
        numerical_cols = [col for col in NUMERICAL_COLS if col in X_train.columns]

    # Add engineered numerical cols
    extra_num = ["charge_per_month_ratio"]
    numerical_cols += [col for col in extra_num if col in X_train.columns]

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    logger.info(f"Scaled numerical columns: {numerical_cols}")
    return X_train, X_test, scaler


# ============================================================
# 6. Split Data
# ============================================================
def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.

    Args:
        df: Encoded DataFrame.
        target_col: Name of the target column.
        test_size: Proportion for test split.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(
        f"Churn ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}"
    )
    return X_train, X_test, y_train, y_test


# ============================================================
# 7. Full Pipeline Orchestrator
# ============================================================
def run_preprocessing(
    data_path: str = None,
    output_dir: str = "data/processed",
) -> dict:
    """
    Execute the full preprocessing pipeline end-to-end.

    Steps:
    1. Load data
    2. Handle missing values
    3. Feature engineering
    4. Encode features
    5. Split data
    6. Scale features
    7. Save processed data and artifacts

    Args:
        data_path: Path to raw CSV file.
        output_dir: Directory to save processed outputs.

    Returns:
        Dictionary with paths to saved artifacts.
    """
    logger.info("=" * 60)
    logger.info("TELCO CHURN PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    start_time = datetime.now()

    # Step 1: Load
    df = load_data(data_path)

    # Step 2: Clean
    df = handle_missing_values(df)

    # Step 3: Feature Engineering
    df = feature_engineering(df)

    # Step 4: Encode
    df, encoders = encode_features(df)

    # Step 5: Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 6: Scale
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Step 7: Save
    os.makedirs(output_dir, exist_ok=True)

    artifacts = {}

    # Save processed data
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    artifacts["X_train"] = os.path.join(output_dir, "X_train.csv")
    artifacts["X_test"] = os.path.join(output_dir, "X_test.csv")
    artifacts["y_train"] = os.path.join(output_dir, "y_train.csv")
    artifacts["y_test"] = os.path.join(output_dir, "y_test.csv")
    logger.info(f"Saved processed data to {output_dir}/")

    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    artifacts["scaler"] = scaler_path
    logger.info(f"Saved scaler to {scaler_path}")

    # Save encoders
    encoders_path = os.path.join(output_dir, "encoders.joblib")
    joblib.dump(encoders, encoders_path)
    artifacts["encoders"] = encoders_path
    logger.info(f"Saved encoders to {encoders_path}")

    # Save feature names
    feature_names = list(X_train.columns)
    features_path = os.path.join(output_dir, "feature_names.json")
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    artifacts["feature_names"] = features_path
    logger.info(f"Saved feature names to {features_path}")

    # Save preprocessing summary
    summary = {
        "timestamp": start_time.isoformat(),
        "raw_rows": int(df.shape[0] + (X_train.shape[0] + X_test.shape[0]) - df.shape[0]),
        "processed_features": len(feature_names),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "churn_ratio_train": float(y_train.mean()),
        "churn_ratio_test": float(y_test.mean()),
        "feature_names": feature_names,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    summary_path = os.path.join(output_dir, "preprocessing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    artifacts["summary"] = summary_path
    logger.info(f"Saved preprocessing summary to {summary_path}")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETED in {elapsed:.2f} seconds")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    return artifacts


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telco Customer Churn - Preprocessing Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw CSV file (downloads from URL if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed outputs (default: data/processed)",
    )
    args = parser.parse_args()

    run_preprocessing(data_path=args.data_path, output_dir=args.output_dir)
