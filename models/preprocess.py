import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_and_split(csv_path: str, test_size: float = 0.2):
    """Load CSV, drop Time, scale Amount, return stratified train/test split."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Time"])

    X = df[FEATURE_COLS].values
    y = df["Class"].values

    # Scale only Amount (index 28) — V1-V28 are already PCA-scaled
    scaler = StandardScaler()
    amount_idx = FEATURE_COLS.index("Amount")
    X[:, amount_idx] = scaler.fit_transform(X[:, amount_idx].reshape(-1, 1)).ravel()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape[0]} samples ({y_train.sum()} fraud)")
    print(f"Test:  {X_test.shape[0]} samples ({y_test.sum()} fraud)")
    print(f"Scaler saved to {SCALER_PATH}")

    return X_train, X_test, y_train, y_test, scaler


def preprocess_single(features: dict, scaler: StandardScaler) -> np.ndarray:
    """Preprocess a single transaction for inference (used by the API)."""
    row = np.array([features[col] for col in FEATURE_COLS], dtype=np.float64)
    amount_idx = FEATURE_COLS.index("Amount")
    row[amount_idx] = scaler.transform([[row[amount_idx]]])[0, 0]
    return row.reshape(1, -1)
