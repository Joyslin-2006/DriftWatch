
import pandas as pd
from pathlib import Path

reference_df = pd.read_csv("data/reference.csv")

output_dir = Path("data/batches")
output_dir.mkdir(parents=True, exist_ok=True)

# Separate legitimate and fraud transactions
legitimate = reference_df[reference_df["Class"] == 0]
fraud = reference_df[reference_df["Class"] == 1]

# Create reproducible batches with approximately the same class ratio
batch_configs = {
    "normal_batch_1": 43,
    "normal_batch_2": 44,
    "normal_batch_3": 45,
    "drifted_batch_1": 101,
    "drifted_batch_2": 102
}

for batch_name, seed in batch_configs.items():

    # Select 995 legitimate and 5 fraud transactions
    legitimate_sample = legitimate.sample(
        n=995, replace=False, random_state=seed
    )

    fraud_sample = fraud.sample(
        n=5, replace=False, random_state=seed
    )

    batch = pd.concat(
        [legitimate_sample, fraud_sample]
    ).sample(frac=1, random_state=seed).copy()

    # Apply controlled Amount shift to drifted batches
    if batch_name == "drifted_batch_1":
        batch["Amount"] *= 2.5

    elif batch_name == "drifted_batch_2":
        batch["Amount"] *= 4

    batch.insert(
        0,
        "transaction_id",
        [f"{batch_name}_{i}" for i in range(len(batch))]
    )

    batch.to_csv(
        output_dir / f"{batch_name}.csv",
        index=False
    )

    print(
        f"{batch_name}: {len(batch)} rows, "
        f"{batch['Class'].sum()} fraud cases"
    )

print("\nAll batches generated successfully!")
