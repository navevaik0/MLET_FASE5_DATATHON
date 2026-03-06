import joblib
import pandas as pd
import numpy as np
import random
import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
router = APIRouter()

# --------------------------------------------------
# CARREGAR MODELO
# --------------------------------------------------

MODEL_PATH = Path("artifacts/model_v1.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

print(f"Modelo carregado com sucesso: {MODEL_PATH}")

# --------------------------------------------------
# CONFIGURAÇÕES
# --------------------------------------------------

EXPECTED_FEATURES = [
    "IAA","IEG","IPS","IDA","IPP","IPV","IAN",
    "DEFASAGEM","NOTA_PORT","NOTA_MAT","NOTA_ING",
    "ANO_INGRESSO","QTDE_AVAL","FASE","TURMA"
]

CATEGORICAL_FEATURES = ["FASE","TURMA"]

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)

MONITORING_PATH = Path("monitoring")
MONITORING_PATH.mkdir(exist_ok=True)

PREDICTION_LOG = LOG_PATH / "predictions.json"

REFERENCE_DATA = MONITORING_PATH / "reference.csv"
PRODUCTION_DATA = MONITORING_PATH / "production.csv"
DRIFT_REPORT = MONITORING_PATH / "drift_report.html"

metrics = {
    "predictions_made": 0,
    "avg_probability": 0
}

start_time = datetime.now()

# --------------------------------------------------
# SCHEMA
# --------------------------------------------------

class StudentInput(BaseModel):

    IAA: float
    IEG: float
    IPS: float
    IDA: float
    IPP: float
    IPV: float
    IAN: float
    DEFASAGEM: float
    NOTA_PORT: float
    NOTA_MAT: float
    NOTA_ING: float
    ANO_INGRESSO: int
    QTDE_AVAL: int
    FASE: str
    TURMA: str

    class Config:
        json_schema_extra = {
            "example": {
                "IAA": 7.5,
                "IEG": 6.8,
                "IPS": 7.1,
                "IDA": 6.9,
                "IPP": 7.0,
                "IPV": 6.5,
                "IAN": 7.2,
                "DEFASAGEM": 0,
                "NOTA_PORT": 8,
                "NOTA_MAT": 7,
                "NOTA_ING": 7,
                "ANO_INGRESSO": 2022,
                "QTDE_AVAL": 4,
                "FASE": "2",
                "TURMA": "A"
            }
        }

# --------------------------------------------------
# FUNÇÕES AUXILIARES
# --------------------------------------------------

def log_prediction(data, prediction, probability):

    record = {
        "timestamp": str(datetime.now()),
        "input": data,
        "prediction": prediction,
        "probability": probability
    }

    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def update_metrics(probability):

    metrics["predictions_made"] += 1

    n = metrics["predictions_made"]
    old_avg = metrics["avg_probability"]

    metrics["avg_probability"] = ((old_avg*(n-1)) + probability)/n


def save_production_data(payload):

    df = pd.DataFrame([payload])

    if not PRODUCTION_DATA.exists():
        df.to_csv(PRODUCTION_DATA, index=False)
    else:
        df.to_csv(PRODUCTION_DATA, mode="a", header=False, index=False)

# --------------------------------------------------
# HEALTHCHECK
# --------------------------------------------------

@router.get(
    "/health",
    summary="Healthcheck da API",
    description="Verifica se a API e o modelo estão funcionando corretamente."
)
def health():

    uptime = datetime.now() - start_time

    return {
        "status": "ok",
        "model_loaded": True,
        "uptime": str(uptime)
    }

# --------------------------------------------------
# INFORMAÇÕES DO MODELO
# --------------------------------------------------

@router.get(
    "/model-info",
    summary="Informações do modelo",
    description="Retorna informações básicas do modelo carregado."
)
def model_info():

    return {
        "model_type": type(model).__name__,
        "features_expected": EXPECTED_FEATURES
    }

# --------------------------------------------------
# SCHEMA DO MODELO
# --------------------------------------------------

@router.get(
    "/model-schema",
    summary="Schema de entrada",
    description="Mostra quais variáveis o modelo espera receber."
)
def model_schema():

    return {
        "expected_features": EXPECTED_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    }

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

@router.post(
    "/predict",
    summary="Predição individual",
    description="Realiza a predição do modelo para um único aluno."
)
def predict(data: StudentInput):

    try:

        payload = data.model_dump()

        df = pd.DataFrame([payload])
        df = df[EXPECTED_FEATURES]

        probability = model.predict_proba(df)[0][1]
        prediction = int(probability >= 0.5)

        log_prediction(payload, prediction, probability)
        update_metrics(probability)
        save_production_data(payload)

        return {
            "prediction": prediction,
            "probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# BATCH PREDICT
# --------------------------------------------------

@router.post(
    "/batch-predict",
    summary="Predição em lote",
    description="Realiza predição para vários alunos ao mesmo tempo."
)
def batch_predict(data: list[StudentInput]):

    try:

        payload = [d.model_dump() for d in data]

        df = pd.DataFrame(payload)
        df = df[EXPECTED_FEATURES]

        probabilities = model.predict_proba(df)[:,1]
        predictions = (probabilities >= 0.5).astype(int)

        results = []

        for i in range(len(df)):
            results.append({
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i])
            })

        return {
            "total_predictions": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# EXPLAIN
# --------------------------------------------------
@router.post(
    "/explain",
    summary="Explicação da decisão do modelo",
    description="""
Mostra quais variáveis mais influenciaram a decisão do modelo para um aluno.

Retorna:
- prediction → classe prevista
- probability → probabilidade da classe positiva
- top_factors → variáveis que mais impactaram a decisão
"""
)
def explain(data: StudentInput):

    try:

        payload = data.model_dump()

        df = pd.DataFrame([payload])
        df = df[EXPECTED_FEATURES]

        # -------------------------
        # PREDIÇÃO
        # -------------------------

        probability = model.predict_proba(df)[0][1]
        prediction = int(probability >= 0.5)

        # -------------------------
        # PREPROCESSAMENTO
        # -------------------------

        preprocessor = model.steps[0][1]
        X_processed = preprocessor.transform(df)

        # nomes das features após transformação
        feature_names = preprocessor.get_feature_names_out()

        # -------------------------
        # COEFICIENTES DO MODELO
        # -------------------------

        classifier = model.steps[-1][1]
        coef = classifier.coef_[0]

        impacts = X_processed[0] * coef

        feature_impacts = []

        for i, impact in enumerate(impacts):

            clean_name = feature_names[i].replace("num__", "").replace("cat__", "")

            feature_impacts.append({
                "feature_index": i,
                "feature": clean_name,
                "impact": float(impact)
            })

        # ordena pelo impacto absoluto
        feature_impacts = sorted(
            feature_impacts,
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        top = feature_impacts[:5]

        return {
            "prediction": prediction,
            "probability": float(probability),
            "top_factors": top
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
@router.get(
    "/feature-importance",
    summary="Importância das variáveis",
    description="Mostra quais variáveis são mais importantes para o modelo."
)
def feature_importance():

    try:

        # pega preprocessador
        preprocessor = model.steps[0][1]

        # pega nomes das features transformadas
        feature_names = preprocessor.get_feature_names_out()

        # pega modelo final
        classifier = model.steps[-1][1]

        # coeficientes da regressão
        coef = classifier.coef_[0]

        importance = []

        for i, value in enumerate(coef):

            clean_name = feature_names[i].replace("num__", "").replace("cat__", "")

            # melhora visual para variáveis categóricas
            if "_" in clean_name:
                parts = clean_name.split("_", 1)
                clean_name = f"{parts[0]} = {parts[1]}"

            importance.append({
                "feature": clean_name,
                "importance": float(abs(value))
            })

        # ordena do mais importante
        importance = sorted(
            importance,
            key=lambda x: x["importance"],
            reverse=True
        )

        return {
            "model": type(classifier).__name__,
            "top_features": importance[:10]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# METRICS
# --------------------------------------------------

@router.get(
    "/metrics",
    summary="Métricas da API",
    description="Mostra estatísticas de uso da API."
)
def get_metrics():

    uptime = datetime.now() - start_time

    return {
        "predictions_made": metrics["predictions_made"],
        "avg_probability": metrics["avg_probability"],
        "uptime": str(uptime)
    }

# --------------------------------------------------
# SIMULAÇÃO
# --------------------------------------------------

@router.get(
    "/simulate-student",
    summary="Simulação de aluno",
    description="Gera um aluno aleatório e executa uma predição."
)
def simulate_student():

    student = {
        "IAA": random.uniform(0,10),
        "IEG": random.uniform(0,10),
        "IPS": random.uniform(0,10),
        "IDA": random.uniform(0,10),
        "IPP": random.uniform(0,10),
        "IPV": random.uniform(0,10),
        "IAN": random.uniform(0,10),
        "DEFASAGEM": random.randint(0,1),
        "NOTA_PORT": random.uniform(0,10),
        "NOTA_MAT": random.uniform(0,10),
        "NOTA_ING": random.uniform(0,10),
        "ANO_INGRESSO": random.randint(2018,2024),
        "QTDE_AVAL": random.randint(1,6),
        "FASE": random.choice(["1","2","3"]),
        "TURMA": random.choice(["A","B","C"])
    }

    df = pd.DataFrame([student])
    df = df[EXPECTED_FEATURES]

    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= 0.5)

    return {
        "generated_student": student,
        "prediction": prediction,
        "probability": probability
    }
