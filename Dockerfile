FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copiar requirements primeiro
COPY requirements.txt .

# atualizar pip e instalar deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copiar projeto
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]