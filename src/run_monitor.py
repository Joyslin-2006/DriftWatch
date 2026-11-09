
import pandas as pd
import joblib
import json


from data_quality import check_data_quality
from drift_detection import detect_feature_drift
from prediction_drift import detect_prediction_drift
from performance_monitor import (
    evaluate_performance,
    compare_with_baseline
)
from alert_engine import generate_alerts


# Convert NumPy values into standard Python values for JSON
def convert_numpy_types(obj):
    if hasattr(obj, "item"):
        return obj.item()

    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


# 1. Load model and reference data
model = joblib.load("models/fraud_model.pkl")

reference_df = pd.read_csv("data/reference.csv")
X_reference = reference_df.drop(columns=["Class"])

reference_predictions = pd.read_csv(
    "data/reference_predictions.csv"
)


# 2. Load incoming batch
batch_id = "normal_batch_1"

batch = pd.read_csv(
    f"data/batches/{batch_id}.csv"
)

X_batch = batch.drop(columns=["transaction_id", "Class"])


# 3. Data quality monitoring
quality_results = check_data_quality(
    X_reference, X_batch
)


# 4. Feature drift monitoring
feature_drift_results = detect_feature_drift(
    X_reference, X_batch
)


# 5. Generate current predictions
current_predictions = pd.DataFrame({
    "transaction_id": batch["transaction_id"],
    "prediction": model.predict(X_batch),
    "fraud_probability": model.predict_proba(X_batch)[:, 1]
})


# 6. Prediction drift monitoring
prediction_drift_results = detect_prediction_drift(
    reference_predictions,
    current_predictions.drop(columns=["transaction_id"])
)


# 7. Delayed-label performance monitoring
labels = pd.read_csv(
    f"data/batches/{batch_id}_labels.csv"
)

performance_results = evaluate_performance(
    current_predictions,
    labels
)

performance_comparison = compare_with_baseline(
    performance_results
)


# 8. Generate alerts
alert_results = generate_alerts(
    quality_results,
    feature_drift_results,
    prediction_drift_results,
    performance_comparison
)


# 9. Display monitoring results
print("\n========== DRIFTWATCH MONITORING ==========")

print("\nDATA QUALITY:")
print(quality_results)

print("\nFEATURE DRIFT:")
print(
    feature_drift_results.sort_values(
        "psi", ascending=False
    ).head(5)
)

print("\nPREDICTION DRIFT:")
print(prediction_drift_results)

print("\nPERFORMANCE:")
print(performance_comparison)

print("\nALERTS:")
print(json.dumps(
    alert_results,
    indent=4,
    default=convert_numpy_types
))


# 10. Save monitoring report
report = {
    "batch_id": batch_id,
    "data_quality": quality_results,
    "feature_drift": feature_drift_results.to_dict(orient="records"),
    "prediction_drift": prediction_drift_results,
    "performance": performance_results,
    "performance_comparison": performance_comparison,
    "alerts": alert_results
}

with open(
    f"reports/{batch_id}_monitoring_report.json",
    "w"
) as file:
    json.dump(
        report,
        file,
        indent=4,
        default=convert_numpy_types
    )

print("\nMonitoring report saved successfully!")
