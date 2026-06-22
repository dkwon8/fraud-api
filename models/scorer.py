from dataclasses import dataclass
import numpy as np
import mlflow
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

IF_WEIGHT = 0.4
AE_WEIGHT = 0.6
THRESHOLD = 0.3


@dataclass
class ScoreResult:
    """Result from scoring a single transaction."""
    if_score: float
    ae_score: float
    final_score: float
    predicted_label: int


def fuse_scores(if_score: float, ae_score: float, threshold: float = THRESHOLD) -> ScoreResult:
    """Combine IF and AE scores into a single fraud score."""
    final = IF_WEIGHT * if_score + AE_WEIGHT * ae_score
    return ScoreResult(
        if_score=if_score,
        ae_score=ae_score,
        final_score=final,
        predicted_label=int(final >= threshold),
    )


def evaluate_ensemble(if_scores: np.ndarray, ae_scores: np.ndarray,
                      y_true: np.ndarray, threshold: float = THRESHOLD):
    """Evaluate the combined model on the test set and log to MLflow."""
    final_scores = IF_WEIGHT * if_scores + AE_WEIGHT * ae_scores
    preds = (final_scores >= threshold).astype(int)

    auc = roc_auc_score(y_true, final_scores)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    with mlflow.start_run(run_name="ensemble"):
        mlflow.log_params({
            "if_weight": IF_WEIGHT,
            "ae_weight": AE_WEIGHT,
            "threshold": threshold,
        })
        mlflow.log_metrics({
            "ensemble_roc_auc": auc,
            "ensemble_precision": precision,
            "ensemble_recall": recall,
            "ensemble_f1": f1,
            "ensemble_fpr": fpr,
        })

    print(f"\n{'='*40}")
    print(f"Ensemble Results (threshold={threshold})")
    print(f"{'='*40}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.6f}")
    print(f"Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return {
        "roc_auc": auc, "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
