from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="PEDE ML API",
    description="API de predição de defasagem educacional",
    version="1.0.0"
)

# registra rotas
app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "PEDE Machine Learning API",
        "status": "running"
    }