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
from src.utils import *
from src.feature_engineering import *
router = APIRouter()

# --------------------------------------------------
# CARREGAR MODELO
# --------------------------------------------------

MODEL_PATH = Path("app/model/model_champion.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

print(f"Modelo carregado com sucesso: {MODEL_PATH}")

# --------------------------------------------------
# CONFIGURAÇÕES
# --------------------------------------------------

artifacts_dir = Path("./artifacts")

#Carrega as variáveis finalistas
numeric_cols = load_list(artifacts_dir / "numeric_final.json")
categorical_cols = load_list(artifacts_dir / "categorical_final.json")

expected_features = numeric_cols + categorical_cols

categorical_cols = categorical_cols

LOG_PATH = Path("./logs")
LOG_PATH.mkdir(exist_ok=True)

MONITORING_PATH = Path("./monitoring")
MONITORING_PATH.mkdir(exist_ok=True)

PREDICTION_LOG = LOG_PATH / "predictions.json"

REFERENCE_DATA = MONITORING_PATH / "reference.csv"
PRODUCTION_DATA = MONITORING_PATH / "production.csv"
PRODUCTION_SAMPLE_JSON = MONITORING_PATH / "production_students.json"
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

    IDADE_ALUNO: int
    ANO_INGRESSO: int
    CG: float
    CF: float
    CT: float
    QTDE_AVAL: int
    IAA: float
    IEG: float
    IPS: float
    NOTA_MAT: float
    NOTA_PORT: float
    NOTA_ING: float
    IPV: float
    IPP: float
    PEDRA_ORDINAL: int
    NOTA_MEDIA: float
    TEMPO_PM: int
    FASE: str
    TURMA: str

    class Config:
        json_schema_extra = {
            "example": {
                "IDADE_ALUNO": 20,
                "ANO_INGRESSO": 2022,
                "CG": 8.5,
                "CF": 7.8,
                "CT": 8.2,
                "QTDE_AVAL": 4,
                "IAA": 7.5,
                "IEG": 6.8,
                "IPS": 7.1,
                "NOTA_MAT": 7.0,
                "NOTA_PORT": 8.0,
                "NOTA_ING": 7.0,
                "IPV": 6.5,
                "IPP": 7.0,
                "PEDRA_ORDINAL": 1,
                "NOTA_MEDIA": 7.5,
                "TEMPO_PM": 5,
                "FASE": "2",
                "TURMA": "A"
            }
        }


class StudentInputBatch(BaseModel):

    IDADE_ALUNO: int
    ANO_INGRESSO: int
    CG: float
    CF: float
    CT: float
    QTDE_AVAL: int
    IAA: float
    IEG: float
    IPS: float
    NOTA_MAT: float
    NOTA_PORT: float
    NOTA_ING: float
    IPV: float
    IPP: float
    PEDRA_ORDINAL: int
    NOTA_MEDIA: float
    TEMPO_PM: int
    FASE: str
    TURMA: str
        

# --------------------------------------------------
# FUNÇÕES AUXILIARES
# --------------------------------------------------

def log_prediction(data, prediction, probability):

    record = {
        "timestamp": str(datetime.now()),
        "input": data,
        "prediction": int(prediction),
        "probability": float(probability)
    }

    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def update_metrics(probability):

    probability = float(probability)

    metrics["predictions_made"] += 1

    n = metrics["predictions_made"]
    old_avg = metrics["avg_probability"]

    metrics["avg_probability"] = ((old_avg*(n-1)) + probability)/n


def save_production_data(payload):

    df = pd.DataFrame([payload])

    # escreve cabeçalho se arquivo não existir ou estiver vazio
    if not PRODUCTION_DATA.exists() or PRODUCTION_DATA.stat().st_size == 0:
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
        "features_expected": expected_features
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
        "expected_features": expected_features,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols
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
        df = df[expected_features]

        #Aplica os binnings
        df_model = apply_numeric_binning(df, load_dict(artifacts_dir / "binning_dict.json"))

        probability = model.predict_proba(df_model)[0][1].item()
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
from fastapi import Body

@router.post(
    "/batch-predict",
    summary="Predição em lote",
    description="Realiza predição para vários alunos ao mesmo tempo."
)
def batch_predict(
    data: list[StudentInputBatch] = Body(
        examples=[
            [
                {
                    "IDADE_ALUNO": 20,
                    "ANO_INGRESSO": 2022,
                    "CG": 8.5,
                    "CF": 7.8,
                    "CT": 8.2,
                    "QTDE_AVAL": 4,
                    "IAA": 7.5,
                    "IEG": 6.8,
                    "IPS": 7.1,
                    "NOTA_MAT": 7.0,
                    "NOTA_PORT": 8.0,
                    "NOTA_ING": 7.0,
                    "IPV": 6.5,
                    "IPP": 7.0,
                    "PEDRA_ORDINAL": 1,
                    "NOTA_MEDIA": 7.5,
                    "TEMPO_PM": 5,
                    "FASE": "2",
                    "TURMA": "A"
                },
                {
                    "IDADE_ALUNO": 13,
                    "ANO_INGRESSO": 2023,
                    "CG": 9.5,
                    "CF": 3.8,
                    "CT": 4.2,
                    "QTDE_AVAL": 8,
                    "IAA": 1.5,
                    "IEG": 5.8,
                    "IPS": 3.1,
                    "NOTA_MAT": 6.0,
                    "NOTA_PORT": 7.0,
                    "NOTA_ING": 6.0,
                    "IPV": 3.5,
                    "IPP": 4.0,
                    "PEDRA_ORDINAL": 2,
                    "NOTA_MEDIA": 3.5,
                    "TEMPO_PM": 4,
                    "FASE": "3",
                    "TURMA": "B"
                }
            ]
        ]
    )
):

    try:

        payload = [d.model_dump() for d in data]

        df = pd.DataFrame(payload)
        df = df[expected_features]

        # Aplica os binnings
        df_model = apply_numeric_binning(df, load_dict(artifacts_dir / "binning_dict.json"))

        probabilities = model.predict_proba(df_model)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results = []

        for i in range(len(df)):

            prediction = int(predictions[i])
            probability = float(probabilities[i])
            student_input = payload[i]

            log_prediction(student_input, prediction, probability)
            update_metrics(probability)
            save_production_data(student_input)

            results.append({
                "prediction": prediction,
                "probability": probability
            })

        return {
            "total_predictions": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    "IDADE_ALUNO": random.randint(11, 22),
    "ANO_INGRESSO": random.randint(2018, 2024),
    "CG": random.uniform(0, 10),
    "CF": random.uniform(0, 10),
    "CT": random.uniform(0, 10),
    "QTDE_AVAL": random.randint(1, 6),
    "IAA": random.uniform(0, 10),
    "IEG": random.uniform(0, 10),
    "IPS": random.uniform(0, 10),
    "NOTA_MAT": random.uniform(0, 10),
    "NOTA_PORT": random.uniform(0, 10),
    "NOTA_ING": random.uniform(0, 10),
    "IPV": random.uniform(0, 10),
    "IPP": random.uniform(0, 10),
    "PEDRA_ORDINAL": random.randint(1, 4),
    "NOTA_MEDIA": random.uniform(0, 10),
    "TEMPO_PM": random.randint(0, 10),
    "FASE": random.choice(["1", "2", "3"]),
    "TURMA": random.choice(["A", "B", "C"])
    }

    df = pd.DataFrame([student])
    df = df[expected_features]

    #Aplica os binnings
    df_model = apply_numeric_binning(df, load_dict(artifacts_dir / "binning_dict.json"))

    probability = model.predict_proba(df_model)[0][1].item()
    prediction = int(probability >= 0.5)

    log_prediction(student, prediction, probability)
    update_metrics(probability)

    return {
        "generated_student": student,
        "prediction": int(prediction),
        "probability": float(probability)
    }
import json


# --------------------------------------------------
# 100-casos
# --------------------------------------------------

@router.get(
    "/predict-amostra_100_casos",
    summary="Predição da amostra de 100 alunos",
    description="Lê o arquivo production_students.json e executa predições para todos os alunos"
)
def predict_production():

    try:

        if not PRODUCTION_SAMPLE_JSON.exists():
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")

        if PRODUCTION_SAMPLE_JSON.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="production_students.json está vazio"
            )

        with open(PRODUCTION_SAMPLE_JSON, "r", encoding="utf-8") as f:
            students = json.load(f)

        df = pd.DataFrame(students)

        df = df[expected_features]

        df_model = apply_numeric_binning(
            df,
            load_dict(artifacts_dir / "binning_dict.json")
        )

        probabilities = model.predict_proba(df_model)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results = []

        for i in range(len(df)):

            payload = df.iloc[i].to_dict()
            prediction = int(predictions[i])
            probability = float(probabilities[i])

            log_prediction(payload, prediction, probability)
            update_metrics(probability)
            save_production_data(payload)   

            results.append({
                "prediction": prediction,
                "probability": probability
            })

        return {
            "total_students": len(df),
            "results": results[:100]
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