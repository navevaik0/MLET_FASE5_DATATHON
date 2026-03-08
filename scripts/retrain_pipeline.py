import json
import pandas as pd
from pathlib import Path

from src.retraining import should_retrain
from scripts.train_pipeline import main as train_pipeline


def main():

    report_path = Path("monitoring/monitoring_report.json")

    if not report_path.exists():
        print("Nenhum relatório de monitoramento encontrado.")
        return

    with open(report_path) as f:
        report = json.load(f)

    drift_report = report["data_drift"]
    prediction_drift = report["prediction_drift"]

    if should_retrain(drift_report, prediction_drift):

        print("Drift detectado. Retreinando modelo")

        train_pipeline()

    else:

        print("Modelo ainda está estável.")


if __name__ == "__main__":
    main()