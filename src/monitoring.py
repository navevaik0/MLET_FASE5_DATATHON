import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def log_metrics(metrics_dict: dict, log_path: str = "monitoring/metrics_log.json"):

    Path("monitoring").mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics_dict
    }

    if Path(log_path).exists():
        with open(log_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(log_path, "w") as f:
        json.dump(data, f, indent=4)


def detect_numeric_drift(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    numeric_features: list,
    threshold: float = 0.2
):
    drift_report = {}

    for col in numeric_features:
        train_mean = train_df[col].mean()
        new_mean = new_df[col].mean()

        relative_change = abs(new_mean - train_mean) / (abs(train_mean) + 1e-6)

        drift_report[col] = {
            "train_mean": train_mean,
            "new_mean": new_mean,
            "relative_change": relative_change,
            "drift_detected": relative_change > threshold
        }

    return drift_report

def detect_prediction_drift(
    train_preds: pd.Series,
    new_preds: pd.Series,
    threshold: float = 0.15
):

    train_rate = train_preds.mean()
    new_rate = new_preds.mean()

    change = abs(new_rate - train_rate)

    return {
        "train_positive_rate": train_rate,
        "new_positive_rate": new_rate,
        "absolute_change": change,
        "drift_detected": change > threshold
    }