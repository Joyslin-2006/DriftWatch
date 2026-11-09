import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def detect_prediction_drift(
    reference_predictions,
    current_predictions
):

    # Fraud prediction percentages
    reference_fraud_rate = (
        reference_predictions["prediction"].mean()
    )

    current_fraud_rate = (
        current_predictions["prediction"].mean()
    )

    fraud_rate_change = (
        current_fraud_rate - reference_fraud_rate
    )

    # Compare fraud probability distributions
    ks_statistic, p_value = ks_2samp(
        reference_predictions["fraud_probability"],
        current_predictions["fraud_probability"]
    )

    return {
        "reference_fraud_rate": float(reference_fraud_rate),
        "current_fraud_rate": float(current_fraud_rate),
        "fraud_rate_change": float(fraud_rate_change),
        "probability_ks_statistic": float(ks_statistic),
        "probability_p_value": float(p_value),
        "prediction_drift_detected": (
            abs(fraud_rate_change) >= 0.01
            or (
                ks_statistic >= 0.1
                and p_value < 0.05
            )
        )
    }