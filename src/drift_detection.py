import numpy as np
import pandas as pd
from scipy.stats import ks_2samp



def calculate_psi(reference, current, bins=10):

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Calculate breakpoints using reference percentiles
    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    # Remove repeated breakpoints
    breakpoints = np.unique(breakpoints)

    # Include values outside the reference range
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Count observations in each bin
    reference_counts, _ = np.histogram(
        reference, bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current, bins=breakpoints
    )

    # Convert counts to proportions
    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 0.0001

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)



def detect_feature_drift(reference_df, current_df):

    results = []

    for column in reference_df.columns:

        reference = reference_df[column].dropna()
        current = current_df[column].dropna()

        # PSI
        psi = calculate_psi(reference, current)

        # KS test
        ks_statistic, p_value = ks_2samp(
            reference, current
        )

        results.append({
            "feature": column,
            "psi": psi,
            "ks_statistic": float(ks_statistic),
            "p_value": float(p_value),
            "drift_detected": (
                psi >= 0.1
                and ks_statistic >= 0.1
                and p_value < 0.05
            )
        })

    return pd.DataFrame(results)