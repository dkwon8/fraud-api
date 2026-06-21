import os
import numpy as np
import joblib
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
IF_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib")


def score_if(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Score a batch: flip sign of decision_function, normalize to [0,1]."""
    raw = -model.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)


def train_isolation_forest(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray):
    X_train_normal = X_train[y_train == 0]

    model = IsolationForest(
        contamination=0.001, n_estimators=200, random_state=42, n_jobs=-1
    )

    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.log_params({
            "model": "IsolationForest",
            "contamination": 0.001,
            "n_estimators": 200,
            "train_normal_samples": len(X_train_normal),
        })

        model.fit(X_train_normal)

        test_scores = score_if(model, X_test)
        auc = roc_auc_score(y_test, test_scores)

        mlflow.log_metric("roc_auc", auc)
        print(f"Isolation Forest ROC-AUC: {auc:.4f}")

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(model, IF_MODEL_PATH)
        mlflow.log_artifact(IF_MODEL_PATH)

    return model, auc
