
import json
from pathlib import Path

import joblib
import pandas as pd

from data_quality import check_data_quality
from drift_detection import detect_feature_drift
from prediction_drift import detect_prediction_drift
from performance_monitor import evaluate_performance, compare_with_baseline
from alert_engine import generate_alerts


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def convert_numpy_types(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )

def run_monitoring(batch_df, batch_id="uploaded_batch", labels_df=None):
    """
    Monitor one incoming batch.

    batch_df: transaction_id + model features; no Class column.
    labels_df: optional transaction_id + actual_class.
    """

    # 1. Load reference data and trained model
    model = joblib.load(PROJECT_ROOT / "models" / "fraud_model.pkl")
    reference_df = pd.read_csv(PROJECT_ROOT / "data" / "reference.csv")
    X_reference = reference_df.drop(columns=["Class"])

    # 2. Separate transaction IDs from model features
    if "transaction_id" not in batch_df.columns:
        raise ValueError("Incoming data must contain a transaction_id column.")

    if "Class" in batch_df.columns or "actual_class" in batch_df.columns:
        raise ValueError(
            "Incoming data must not contain actual labels. "
            "Keep labels in a separate file."
        )

    transaction_ids = batch_df["transaction_id"].copy()
    X_batch = batch_df.drop(columns=["transaction_id"]).copy()

    # 3. Validate feature schema
    if set(X_batch.columns) != set(X_reference.columns):
        missing = sorted(set(X_reference.columns) - set(X_batch.columns))
        extra = sorted(set(X_batch.columns) - set(X_reference.columns))
        raise ValueError(
            f"Feature columns do not match reference data. "
            f"Missing: {missing}; extra: {extra}"
        )

    # Keep the exact reference feature order
    X_batch = X_batch[X_reference.columns]

    # 4. Data quality monitoring
    quality_results = check_data_quality(X_reference, X_batch)

    if quality_results["status"] != "PASS":
        raise ValueError(
            f"Data quality checks failed. Resolve these issues first: "
            f"{quality_results['issues']}"
        )

    # 5. Feature drift monitoring
    feature_drift_results = detect_feature_drift(X_reference, X_batch)

    # 6. Generate predictions
    current_predictions = pd.DataFrame({
        "transaction_id": transaction_ids,
        "prediction": model.predict(X_batch),
        "fraud_probability": model.predict_proba(X_batch)[:, 1]
    })

    # 7. Prediction drift monitoring
    reference_predictions = pd.read_csv(
        PROJECT_ROOT / "data" / "reference_predictions.csv"
    )

    prediction_drift_results = detect_prediction_drift(
        reference_predictions,
        current_predictions.drop(columns=["transaction_id"])
    )

    # 8. Performance monitoring (only when labels are available)
    performance_results = None
    performance_comparison = None
    label_status = "PENDING"

    if labels_df is not None:
        required_label_columns = {"transaction_id", "actual_class"}
        if not required_label_columns.issubset(labels_df.columns):
            raise ValueError(
                "Labels must contain transaction_id and actual_class."
            )

        if labels_df["transaction_id"].duplicated().any():
            raise ValueError("Labels contain duplicate transaction IDs.")

        performance_results = evaluate_performance(
            current_predictions, labels_df
        )
        performance_comparison = compare_with_baseline(performance_results)
        label_status = "AVAILABLE"

    # 9. Generate alerts
    alert_results = generate_alerts(
        quality_results,
        feature_drift_results,
        prediction_drift_results,
        performance_comparison
    )

    # 10. Save predictions and monitoring report
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = reports_dir / f"{batch_id}_predictions.csv"
    current_predictions.to_csv(predictions_path, index=False)

    report = {
        "batch_id": batch_id,
        "label_status": label_status,
        "data_quality": quality_results,
        "feature_drift": feature_drift_results.to_dict(orient="records"),
        "prediction_drift": prediction_drift_results,
        "performance": performance_results,
        "performance_comparison": performance_comparison,
        "alerts": alert_results
    }

    report_path = reports_dir / f"{batch_id}_monitoring_report.json"

    with open(report_path, "w") as file:
        json.dump(
            report,
            file,
            indent=4,
            default=convert_numpy_types
        )

    return report
