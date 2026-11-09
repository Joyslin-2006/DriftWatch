
import json
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# 1. Load trained model
model = joblib.load("models/fraud_model.pkl")

# 2. Load test and reference datasets
test_df = pd.read_csv("data/test.csv")
reference_df = pd.read_csv("data/reference.csv")

# 3. Separate test features and labels
X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

# 4. Calculate baseline performance
test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)[:, 1]

baseline_metrics = {
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
    "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
    "f1_score": float(f1_score(y_test, test_predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "pr_auc": float(average_precision_score(y_test, test_probabilities)),
    "confusion_matrix": confusion_matrix(
        y_test, test_predictions
    ).tolist(),
    "test_rows": int(len(X_test))
}

# 5. Save baseline metrics
with open("reports/baseline_metrics.json", "w") as file:
    json.dump(baseline_metrics, file, indent=4)

# 6. Generate predictions for reference dataset
X_reference = reference_df.drop(columns=["Class"])
y_reference = reference_df["Class"]

reference_predictions = model.predict(X_reference)
reference_probabilities = model.predict_proba(X_reference)[:, 1]

# 7. Save predictions separately
prediction_df = pd.DataFrame({
    "transaction_id": range(len(X_reference)),
    "prediction": reference_predictions,
    "fraud_probability": reference_probabilities
})

prediction_df.to_csv(
    "data/reference_predictions.csv", index=False
)

# 8. Save actual labels separately
label_df = pd.DataFrame({
    "transaction_id": range(len(X_reference)),
    "actual_class": y_reference.to_numpy()
})

label_df.to_csv(
    "data/reference_labels.csv", index=False
)

print("Baseline metrics saved!")
print("Reference predictions saved!")
print("Reference labels saved separately!")

print("\n--- BASELINE PERFORMANCE ---")
for metric, value in baseline_metrics.items():
    print(metric, ":", value)
