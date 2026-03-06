FROM python:3.12-slim

# evita arquivos desnecessários
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# diretório de trabalho
WORKDIR /app

# copiar requirements primeiro (cache do docker)
COPY requirements.txt .

# instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# copiar projeto inteiro
COPY . .

# expor porta da API
EXPOSE 8000

# iniciar API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]