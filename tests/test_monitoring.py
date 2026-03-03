import pandas as pd
import json
from src.monitoring import *


def test_prediction_drift_sem_drift():
    train_preds = pd.Series([1, 0, 1, 1, 0])
    new_preds = pd.Series([1, 0, 1, 0, 1])

    result = detect_prediction_drift(
        train_preds,
        new_preds,
        threshold=0.5
    )

    assert result["drift_detected"] == False
    assert "absolute_change" in result

def test_prediction_drift_com_drift():
    train_preds = pd.Series([1, 1, 1, 1])
    new_preds = pd.Series([0, 0, 0, 0])

    result = detect_prediction_drift(
        train_preds,
        new_preds,
        threshold=0.1
    )

    assert result["drift_detected"] == True


def test_numeric_drift_sem_drift():
    train_df = pd.DataFrame({
        "INDE": [7, 8, 9]
    })

    new_df = pd.DataFrame({
        "INDE": [7.1, 8.1, 8.9]
    })

    result = detect_numeric_drift(
        train_df,
        new_df,
        numeric_features=["INDE"],
        threshold=0.5
    )

    assert result["INDE"]["drift_detected"] == False

def test_numeric_drift_com_drift():
    train_df = pd.DataFrame({
        "INDE": [5, 5, 5]
    })

    new_df = pd.DataFrame({
        "INDE": [10, 10, 10]
    })

    result = detect_numeric_drift(
        train_df,
        new_df,
        numeric_features=["INDE"],
        threshold=0.2
    )

    assert result["INDE"]["drift_detected"] == True


def test_log_metrics(tmp_path):
    log_file = tmp_path / "metrics_log.json"

    metrics = {
        "accuracy": 0.9,
        "roc_auc": 0.85
    }

    log_metrics(metrics, log_path=str(log_file))

    assert log_file.exists()

    with open(log_file, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert "metrics" in data[0]