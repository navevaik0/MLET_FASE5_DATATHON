from datetime import datetime

def should_retrain(
    drift_report: dict,
    prediction_drift: dict,
    drift_threshold: int = 3
):
    """
    Decide se o modelo deve ser retreinado com base em drift.
    """

    numeric_drift_count = sum(
        v["drift_detected"] for v in drift_report.values()
    )

    if numeric_drift_count >= drift_threshold:
        return True

    if prediction_drift["drift_detected"]:
        return True

    return False