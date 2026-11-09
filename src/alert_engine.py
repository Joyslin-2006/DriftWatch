
def generate_alerts(
    data_quality,
    feature_drift,
    prediction_drift,
    performance_comparison=None
):
    alerts = []

    if data_quality["status"] != "PASS":
        alerts.append({
            "type": "DATA_QUALITY",
            "severity": "WARNING",
            "message": "Data quality issues detected."
        })

    drifted_features = feature_drift[
        feature_drift["drift_detected"] == True
    ]["feature"].tolist()

    if drifted_features:
        alerts.append({
            "type": "FEATURE_DRIFT",
            "severity": "WARNING",
            "message": f"Feature drift detected in: {drifted_features}"
        })

    if prediction_drift["prediction_drift_detected"]:
        alerts.append({
            "type": "PREDICTION_DRIFT",
            "severity": "WARNING",
            "message": "Model prediction distribution has changed."
        })

    # Performance alerts are evaluated only when labels are available
    if performance_comparison is not None:
        f1_change = performance_comparison["f1_score"]["change"]
        pr_auc_change = performance_comparison["pr_auc"]["change"]

        if f1_change <= -0.10 or pr_auc_change <= -0.10:
            alerts.append({
                "type": "PERFORMANCE_DEGRADATION",
                "severity": "CRITICAL",
                "message": (
                    "Model performance has dropped significantly "
                    "compared with baseline."
                )
            })

    if any(alert["severity"] == "CRITICAL" for alert in alerts):
        status = "CRITICAL"
    elif alerts:
        status = "WARNING"
    else:
        status = "HEALTHY"

    return {
        "overall_status": status,
        "alert_count": len(alerts),
        "alerts": alerts
    }
