
import pandas as pd
from pathlib import Path

batch_dir = Path("data/batches")

batch_names = [
    "normal_batch_1",
    "normal_batch_2",
    "normal_batch_3",
    "drifted_batch_1",
    "drifted_batch_2"
]

for batch_name in batch_names:

    # Load the generated batch
    batch = pd.read_csv(
        batch_dir / f"{batch_name}.csv"
    )

    # Extract the original labels and matching transaction IDs
    label_df = batch[
        ["transaction_id", "Class"]
    ].rename(
        columns={"Class": "actual_class"}
    )

    # Save delayed labels separately
    label_df.to_csv(
        batch_dir / f"{batch_name}_labels.csv",
        index=False
    )

    print(
        f"{batch_name}: {len(label_df)} labels saved, "
        f"{label_df['actual_class'].sum()} fraud cases"
    )

print("\nAll matching labels generated successfully!")
