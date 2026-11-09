import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_performance(predictions_df, labels_df):

    # Match predictions with their actual labels
    evaluation_df = predictions_df.merge(
        labels_df,
        on="transaction_id"
    )

    y_true = evaluation_df["actual_class"]
    y_pred = evaluation_df["prediction"]
    y_prob = evaluation_df["fraud_probability"]

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_prob)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist(),
        "evaluated_rows": int(len(evaluation_df))
    }


def compare_with_baseline(current_metrics):

    with open("reports/baseline_metrics.json", "r") as file:
        baseline_metrics = json.load(file)

    comparison = {}

    for metric in ["precision", "recall", "f1_score", "pr_auc"]:

        baseline_value = baseline_metrics[metric]
        current_value = current_metrics[metric]

        comparison[metric] = {
            "baseline": baseline_value,
            "current": current_value,
            "change": current_value - baseline_value
        }

    return comparison