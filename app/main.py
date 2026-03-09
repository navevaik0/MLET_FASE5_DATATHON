from fastapi import FastAPI
from pathlib import Path
from app.routes import router

app = FastAPI(
    title="PEDE ML API",
    description="API de predição de defasagem educacional",
    version="1.0.0"
)

# --------------------------------------------------
# LIMPAR production.csv AO INICIAR API
# --------------------------------------------------

PRODUCTION_DATA = Path("./monitoring/production.csv")

@app.on_event("startup")
def clear_production_data():

    if PRODUCTION_DATA.exists():
        PRODUCTION_DATA.unlink()
        print("production.csv removido ao iniciar a API")

    PRODUCTION_DATA.touch()
    print("production.csv criado vazio")

# --------------------------------------------------
# ROTAS
# --------------------------------------------------

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "PEDE Machine Learning API",
        "status": "running"
    }