import pandas as pd
from drift_detection import detect_feature_drift

# Load reference features
reference_df = pd.read_csv("data/reference.csv")
reference_df = reference_df.drop(columns=["Class"])

# Load batches
normal_batch = pd.read_csv("data/batches/normal_batch_1.csv")
drifted_batch = pd.read_csv("data/batches/drifted_batch_1.csv")

# Remove transaction IDs
normal_batch = normal_batch.drop(columns=["transaction_id"])
drifted_batch = drifted_batch.drop(columns=["transaction_id"])

# Test normal batch
print("\n--- NORMAL BATCH DRIFT RESULTS ---")
normal_results = detect_feature_drift(
    reference_df, normal_batch
)
print(normal_results.sort_values("psi", ascending=False).head(10))

# Test drifted batch
print("\n--- DRIFTED BATCH RESULTS ---")
drifted_results = detect_feature_drift(
    reference_df, drifted_batch
)
print(drifted_results.sort_values("psi", ascending=False).head(10))