"""
Compute FPR from labeled transactions in the DB and log to MLflow.

Usage:
    python -m mlflow_tracking.drift
"""
import os
import time

import mlflow
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL_SYNC = os.getenv("DATABASE_URL_SYNC", "postgresql://postgres:password@localhost:5432/frauddb")
BATCH_SIZE = 500


def compute_fpr(cursor, offset: int, limit: int) -> float | None:
    """Compute FPR for a batch of labeled transactions."""
    cursor.execute(
        "SELECT predicted_label, true_label FROM transactions "
        "WHERE true_label IS NOT NULL "
        "ORDER BY scored_at "
        "OFFSET %s LIMIT %s",
        (offset, limit),
    )
    rows = cursor.fetchall()
    if not rows:
        return None

    fp = sum(1 for pred, actual in rows if pred == 1 and actual == 0)
    tn = sum(1 for pred, actual in rows if pred == 0 and actual == 0)

    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)


def run_drift_tracking():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("fraud-detection-drift")

    conn = psycopg2.connect(DATABASE_URL_SYNC)
    cursor = conn.cursor()

    # Count total labeled transactions
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE true_label IS NOT NULL")
    total_labeled = cursor.fetchone()[0]
    print(f"Total labeled transactions: {total_labeled}")

    if total_labeled == 0:
        print("No labeled transactions yet. Run the consumer first.")
        conn.close()
        return

    with mlflow.start_run(run_name="fpr-drift"):
        batch_n = 0
        offset = 0

        while offset < total_labeled:
            fpr = compute_fpr(cursor, offset, BATCH_SIZE)
            if fpr is None:
                break

            mlflow.log_metric("fpr", fpr, step=batch_n)
            print(f"  Batch {batch_n}: offset={offset}, FPR={fpr:.6f}")

            batch_n += 1
            offset += BATCH_SIZE

    conn.close()
    print(f"Logged {batch_n} FPR data points to MLflow.")


def main():
    run_drift_tracking()


if __name__ == "__main__":
    main()
