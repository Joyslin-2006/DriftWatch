
import pandas as pd
from pathlib import Path

# Folder containing our existing generated batches
batch_dir = Path("data/batches")

# Folder where incoming CSV files will be stored
incoming_dir = Path("data/incoming_samples")

# Create the folder if it doesn't exist
incoming_dir.mkdir(parents=True, exist_ok=True)

# Names of the five batches
batch_names = [
    "normal_batch_1",
    "normal_batch_2",
    "normal_batch_3",
    "drifted_batch_1",
    "drifted_batch_2"
]

# Prepare each batch for incoming data
for name in batch_names:

    # Read the original batch
    df = pd.read_csv(batch_dir / f"{name}.csv")

    # Remove actual fraud labels
    incoming_df = df.drop(columns=["Class"])

    # Save the feature-only CSV
    output_path = incoming_dir / f"{name}_incoming.csv"
    incoming_df.to_csv(output_path, index=False)

    print(
        f"{name}: {len(incoming_df)} rows | "
        f"{len(incoming_df.columns)} columns | "
        f"Class column present: {'Class' in incoming_df.columns}"
    )

print("\nAll incoming batches prepared successfully!")
