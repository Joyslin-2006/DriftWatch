import pandas as pd
import joblib

from prediction_drift import detect_prediction_drift

# Load model
model = joblib.load("models/fraud_model.pkl")

# Load reference data
reference_df = pd.read_csv("data/reference.csv")
X_reference = reference_df.drop(columns=["Class"])

# Load one incoming batch
normal_batch = pd.read_csv("data/batches/normal_batch_1.csv")
drifted_batch = pd.read_csv("data/batches/drifted_batch_1.csv")

# Remove transaction IDs
X_normal = normal_batch.drop(columns=["transaction_id"])
X_drifted = drifted_batch.drop(columns=["transaction_id"])

# Load stored reference predictions
reference_predictions = pd.read_csv(
    "data/reference_predictions.csv"
)

# Generate predictions for incoming batches
normal_predictions = pd.DataFrame({
    "prediction": model.predict(X_normal),
    "fraud_probability": model.predict_proba(X_normal)[:, 1]
})

drifted_predictions = pd.DataFrame({
    "prediction": model.predict(X_drifted),
    "fraud_probability": model.predict_proba(X_drifted)[:, 1]
})

# Compare prediction distributions
print("\n--- NORMAL BATCH ---")
print(
    detect_prediction_drift(
        reference_predictions,
        normal_predictions
    )
)

print("\n--- DRIFTED BATCH ---")
print(
    detect_prediction_drift(
        reference_predictions,
        drifted_predictions
    )
)