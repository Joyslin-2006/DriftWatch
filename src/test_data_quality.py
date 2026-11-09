import pandas as pd
from data_quality import check_data_quality

# Load reference data (features only)
reference_df = pd.read_csv("data/reference.csv")
reference_df = reference_df.drop(columns=["Class"])

# Load a normal batch
normal_batch = pd.read_csv("data/batches/normal_batch_1.csv")
normal_batch = normal_batch.drop(columns=["transaction_id"])

# Load a drifted batch
drifted_batch = pd.read_csv("data/batches/drifted_batch_1.csv")
drifted_batch = drifted_batch.drop(columns=["transaction_id"])

# Check both batches
print("\n--- NORMAL BATCH ---")
print(check_data_quality(reference_df, normal_batch))

print("\n--- DRIFTED BATCH ---")
print(check_data_quality(reference_df, drifted_batch))