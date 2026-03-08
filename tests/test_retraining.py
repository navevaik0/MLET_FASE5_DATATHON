from src.retraining import *

def test_should_not_retrain_without_drift():

    drift_report = {
        "feature1": {"drift_detected": False},
        "feature2": {"drift_detected": False}
    }

    prediction_drift = {
        "drift_detected": False
    }

    result = should_retrain(
        drift_report,
        prediction_drift,
        drift_threshold=2
    )

    assert result is False

def test_should_retrain_with_feature_drift():

    drift_report = {
        "feature1": {"drift_detected": True},
        "feature2": {"drift_detected": True},
        "feature3": {"drift_detected": True}
    }

    prediction_drift = {
        "drift_detected": False
    }

    result = should_retrain(
        drift_report,
        prediction_drift,
        drift_threshold=2
    )

    assert result is True