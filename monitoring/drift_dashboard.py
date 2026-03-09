import sys
from pathlib import Path
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --------------------------------------------------
# GARANTE ACESSO AO SRC
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_dict
from src.feature_engineering import apply_numeric_binning

# --------------------------------------------------
# CAMINHOS DO PROJETO
# --------------------------------------------------

artifacts_dir = PROJECT_ROOT / "artifacts"
monitoring_dir = PROJECT_ROOT / "monitoring"

reference_path = monitoring_dir / "reference.csv"
production_path = monitoring_dir / "production.csv"
report_path = monitoring_dir / "drift_report.html"

# --------------------------------------------------
# VALIDA ARQUIVOS
# --------------------------------------------------

if not reference_path.exists():
    raise FileNotFoundError(f"Reference dataset não encontrado: {reference_path}")

if not production_path.exists():
    raise FileNotFoundError(f"Production dataset não encontrado: {production_path}")

# --------------------------------------------------
# CARREGA DATASETS
# --------------------------------------------------

reference = pd.read_csv(reference_path)
production = pd.read_csv(production_path)

print("Reference shape:", reference.shape)
print("Production shape:", production.shape)

# --------------------------------------------------
# REMOVE TARGET SE EXISTIR
# --------------------------------------------------

if "DEFASADO" in reference.columns:
    reference = reference.drop(columns=["DEFASADO"])

if "DEFASADO" in production.columns:
    production = production.drop(columns=["DEFASADO"])

# --------------------------------------------------
# PREPARA FEATURES DO REFERENCE
# --------------------------------------------------
# reference já está pós feature engineering

reference_features = [c for c in reference.columns if c.startswith("CAT_")]

reference = reference[reference_features]

print("Reference features usadas:", len(reference_features))

# --------------------------------------------------
# PREPARA PRODUCTION
# --------------------------------------------------
# production vem RAW da API

binning_dict = load_dict(artifacts_dir / "binning_dict.json")

production_fe = apply_numeric_binning(production, binning_dict)

production_features = [c for c in production_fe.columns if c.startswith("CAT_")]

production_fe = production_fe[production_features]

print("Production features usadas:", len(production_features))

# --------------------------------------------------
# ALINHAR COLUNAS
# --------------------------------------------------

common_cols = list(set(reference.columns) & set(production_fe.columns))

reference = reference[common_cols]
production_fe = production_fe[common_cols]

print("Colunas usadas no drift:", len(common_cols))

# --------------------------------------------------
# CRIA RELATÓRIO
# --------------------------------------------------

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=reference,
    current_data=production_fe
)

# --------------------------------------------------
# SALVA RELATÓRIO
# --------------------------------------------------

report.save_html(str(report_path))

print("\nDrift report gerado em:")
print(report_path)