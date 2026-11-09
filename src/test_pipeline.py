
import pandas as pd
from pathlib import Path
from monitoring_pipeline import run_monitoring

incoming_dir = Path("data/incoming_samples")
labels_dir = Path("data/batches")

batch_names = [
    "normal_batch_1",
    "normal_batch_2",
    "normal_batch_3",
    "drifted_batch_1",
    "drifted_batch_2"
]

for name in batch_names:
    batch = pd.read_csv(incoming_dir / f"{name}_incoming.csv")
    labels = pd.read_csv(labels_dir / f"{name}_labels.csv")

    # First run: simulate monitoring before labels arrive
    pending = run_monitoring(batch, batch_id=name)

    # Second run: evaluate performance after labels arrive
    evaluated = run_monitoring(
        batch,
        batch_id=f"{name}_evaluated",
        labels_df=labels
    )

    print(f"\n{name}")
    print("  Before labels:", pending["label_status"])
    print("  After labels:", evaluated["label_status"])
    print("  Alert status:", evaluated["alerts"]["overall_status"])
    print("  F1 score:", evaluated["performance"]["f1_score"])
