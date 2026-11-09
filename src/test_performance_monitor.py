import pandas as pd
import joblib

from performance_monitor import (
    evaluate_performance,
    compare_with_baseline
)

# 1. Load trained model
model = joblib.load("models/fraud_model.pkl")

# 2. Load one incoming batch
batch = pd.read_csv(
    "data/batches/normal_batch_1.csv"
)

# 3. Separate transaction IDs and features
transaction_ids = batch["transaction_id"]
X_batch = batch.drop(columns=["transaction_id"])

# 4. Generate model predictions
predictions_df = pd.DataFrame({
    "transaction_id": transaction_ids,
    "prediction": model.predict(X_batch),
    "fraud_probability": model.predict_proba(X_batch)[:, 1]
})

# 5. Load delayed actual labels
labels_df = pd.read_csv(
    "data/batches/normal_batch_1_labels.csv"
)

# 6. Evaluate performance
current_metrics = evaluate_performance(
    predictions_df,
    labels_df
)

print("\n--- CURRENT BATCH PERFORMANCE ---")
for metric, value in current_metrics.items():
    print(f"{metric}: {value}")

# 7. Compare against baseline
comparison = compare_with_baseline(current_metrics)

print("\n--- BASELINE COMPARISON ---")
for metric, values in comparison.items():
    print(f"\n{metric}")
    print(f"Baseline: {values['baseline']:.4f}")
    print(f"Current:  {values['current']:.4f}")
    print(f"Change:   {values['change']:+.4f}")