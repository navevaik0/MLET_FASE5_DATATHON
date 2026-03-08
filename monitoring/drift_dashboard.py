import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# carregar datasets
reference = pd.read_csv("./data/processed/reference.csv")
production = pd.read_csv("monitoring/production.csv")

# remover target se existir
if "DEFASADO" in reference.columns:
    reference = reference.drop(columns=["DEFASADO"])

if "DEFASADO" in production.columns:
    production = production.drop(columns=["DEFASADO"])

# garantir mesmas colunas
common_cols = list(set(reference.columns) & set(production.columns))
reference = reference[common_cols]
production = production[common_cols]

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=reference,
    current_data=production
)

report.save_html("monitoring/drift_report.html")

print("Drift report gerado em monitoring/drift_report.html")