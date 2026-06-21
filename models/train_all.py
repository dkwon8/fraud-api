"""
Full training pipeline: preprocess → Isolation Forest → Autoencoder → Ensemble evaluation.

Usage:
    python -m models.train_all
    python -m models.train_all --csv path/to/creditcard.csv
"""
import argparse
import os
import sys
import joblib
import mlflow
from dotenv import load_dotenv

load_dotenv()

from models.preprocess import load_and_split
from models.train_if import train_isolation_forest, score_if
from models.autoencoder import train_autoencoder, reconstruction_error, normalize_scores
from models.scorer import evaluate_ensemble

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/creditcard.csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Place the Kaggle creditcard.csv in data/")
        sys.exit(1)

    # Point MLflow at our local SQLite backend
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("fraud-detection")

    # Step 1: Load and preprocess the data
    print("=" * 50)
    print("Step 1: Preprocessing")
    print("=" * 50)
    X_train, X_test, y_train, y_test, scaler = load_and_split(args.csv)

    # Step 2: Train Isolation Forest on normal transactions only
    print("\n" + "=" * 50)
    print("Step 2: Training Isolation Forest")
    print("=" * 50)
    if_model, if_auc = train_isolation_forest(X_train, y_train, X_test, y_test)

    # Step 3: Train Autoencoder on normal transactions only
    print("\n" + "=" * 50)
    print("Step 3: Training Autoencoder")
    print("=" * 50)
    ae_model, ae_auc, device = train_autoencoder(X_train, y_train, X_test, y_test)

    # Step 4: Evaluate the ensemble (both models combined)
    print("\n" + "=" * 50)
    print("Step 4: Ensemble Evaluation")
    print("=" * 50)
    if_test_scores = score_if(if_model, X_test)
    ae_test_errors = reconstruction_error(ae_model, X_test, device)
    ae_test_scores = normalize_scores(ae_test_errors)
    evaluate_ensemble(if_test_scores, ae_test_scores, y_test)

    # Save normalization bounds for inference
    # The API needs these to normalize raw scores the same way training did
    if_train_raw = -if_model.decision_function(X_train)
    ae_train_errors = reconstruction_error(ae_model, X_train, device)
    bounds = {
        "if_min": float(if_train_raw.min()),
        "if_max": float(if_train_raw.max()),
        "ae_min": float(ae_train_errors.min()),
        "ae_max": float(ae_train_errors.max()),
    }
    joblib.dump(bounds, os.path.join(ARTIFACTS_DIR, "norm_bounds.joblib"))

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print("Training complete.")


if __name__ == "__main__":
    main()
