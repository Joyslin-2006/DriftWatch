
import pandas as pd
import numpy as np


def check_data_quality(reference_df, batch_df):

    issues = []

    # 1. Check whether columns match
    if set(reference_df.columns) != set(batch_df.columns):
        issues.append("Column mismatch")

    # 2. Check for missing values
    missing_values = batch_df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]

    if not missing_columns.empty:
        issues.append(
            f"Missing values found: {missing_columns.to_dict()}"
        )

    # 3. Check for infinite values in numeric columns
    numeric_df = batch_df.select_dtypes(include=[np.number])
    infinite_counts = np.isinf(numeric_df).sum()
    infinite_columns = infinite_counts[infinite_counts > 0]

    if not infinite_columns.empty:
        issues.append(
            f"Infinite values found: {infinite_columns.to_dict()}"
        )

    # 4. Check duplicate rows
    duplicate_count = batch_df.duplicated().sum()

    if duplicate_count > 0:
        issues.append(f"Duplicate rows found: {duplicate_count}")

    # 5. Check data types
    common_columns = set(reference_df.columns) & set(batch_df.columns)

    for column in common_columns:
        if reference_df[column].dtype != batch_df[column].dtype:
            issues.append(
                f"Data type mismatch in column: {column}"
            )

    # Final result
    return {
        "total_rows": int(len(batch_df)),
        "missing_values": int(batch_df.isnull().sum().sum()),
        "duplicate_rows": int(duplicate_count),
        "infinite_values": int(np.isinf(numeric_df).sum().sum()),
        "issues": issues,
        "status": "PASS" if len(issues) == 0 else "WARNING"
    }