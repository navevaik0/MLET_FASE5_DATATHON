Predição de Risco de Defasagem Escolar – Passos Mágicos
Visão Geral do Projeto

Este projeto implementa um sistema completo de Machine Learning para prever o risco de defasagem escolar de estudantes atendidos pela Associação Passos Mágicos. A solução foi desenvolvida seguindo boas práticas de engenharia de Machine Learning e MLOps, incluindo pipeline de dados, treinamento do modelo, disponibilização de uma API de predição, containerização com Docker e monitoramento contínuo de drift.

O objetivo do modelo é identificar alunos com maior risco de defasagem educacional, permitindo intervenções pedagógicas antecipadas e mais eficientes.

Problema de Negócio

A Associação Passos Mágicos atua no apoio educacional de crianças e jovens em situação de vulnerabilidade social. Um dos desafios enfrentados pela instituição é identificar precocemente alunos com risco de atraso ou defasagem no aprendizado.

Este projeto busca responder à seguinte pergunta:

Qual é a probabilidade de um estudante apresentar defasagem educacional?

A resposta permite priorizar suporte pedagógico, psicológico e educacional de forma mais eficiente.

Solução Proposta

A solução implementa uma pipeline completa de Machine Learning composta por:

Pré-processamento e limpeza de dados

Engenharia de atributos

Treinamento e validação do modelo

Serialização do modelo treinado

Disponibilização do modelo por meio de uma API

Containerização com Docker

Monitoramento de drift em produção

Testes automatizados

Stack Tecnológica

Linguagem
Python 3

Bibliotecas de Machine Learning
scikit-learn
pandas
numpy
xgboost

API
FastAPI

Serialização de modelo
joblib

Containerização
Docker

Testes
pytest

Monitoramento
Evidently

Estrutura do Projeto
project-root
│
├── app
│   ├── main.py                # Inicialização da API
│   ├── routes.py              # Endpoints da API
│   └── model
│        ├── model_base.joblib
│        └── model_champion.joblib
│
├── src                        # Código da pipeline de ML
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── evaluate.py
│   └── monitoring.py
│
├── scripts                    # Scripts executáveis da pipeline
│   ├── data_pipeline.py
│   ├── train_pipeline.py
│   └── retrain_pipeline.py
│
├── monitoring                 # Monitoramento do modelo
│   ├── drift_dashboard.py
│   ├── drift_report.html
│   ├── reference.csv
│   └── production.csv
│
├── artifacts                  # Artefatos do modelo
│   ├── binning_dict.json
│   ├── numeric_cols.json
│   ├── categorical_cols.json
│   └── model_metadata.json
│
├── notebooks                  # Análises exploratórias e experimentos
│
├── tests                      # Testes automatizados
│
├── logs
│   └── predictions.json
│
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
Pipeline de Machine Learning

A pipeline do projeto é composta pelas seguintes etapas.

1. Pré-processamento dos Dados

Responsável por:

limpeza de dados

tratamento de valores ausentes

padronização de tipos

preparação para engenharia de features

Arquivo principal:

src/preprocessing.py
2. Engenharia de Features

Nesta etapa são criadas variáveis derivadas que capturam melhor os padrões educacionais do dataset.

Exemplos:

discretização de variáveis numéricas

criação de categorias educacionais

transformação de métricas acadêmicas

Arquivo principal:

src/feature_engineering.py
3. Seleção de Features

Seleção das variáveis mais relevantes para o modelo.

Arquivo:

src/feature_selection.py
4. Treinamento do Modelo

O treinamento é realizado utilizando algoritmos de classificação, com validação e seleção do melhor modelo.

Arquivo principal:

scripts/train_pipeline.py

O modelo final é serializado em:

app/model/model_champion.joblib
5. Avaliação do Modelo

O desempenho do modelo é analisado utilizando métricas de classificação.

Arquivo:

src/evaluate.py
API de Predição

O modelo treinado é disponibilizado por meio de uma API construída com FastAPI.

Endpoints principais:

GET /health
Verifica se a API está ativa.

POST /predict
Realiza predição para um estudante.

POST /batch-predict
Realiza predições para múltiplos estudantes.

GET /feature-importance
Retorna as variáveis mais importantes do modelo.

GET /metrics
Retorna métricas de uso da API.

Documentação interativa da API:

http://localhost:8000/docs
Containerização com Docker

A aplicação é empacotada utilizando Docker para garantir portabilidade e reprodutibilidade do ambiente.

Build da imagem
docker build -t pede-ml-api .
Execução do container
docker run -p 8000:8000 pede-ml-api

A API ficará disponível em:

http://localhost:8000
Monitoramento do Modelo

O projeto implementa monitoramento de drift de dados utilizando a biblioteca Evidently.

Arquivo principal:

monitoring/drift_dashboard.py

Esse script compara:

dataset de referência (dados de treinamento)

dataset de produção

e gera um relatório de drift.

Execução:

python monitoring/drift_dashboard.py

O relatório é gerado em:

monitoring/drift_report.html
Testes Automatizados

O projeto possui testes unitários para validar diferentes componentes da pipeline.

Diretório:

tests/

Execução:

pytest
Execução Completa do Projeto
1 Instalar dependências
pip install -r requirements.txt

Para ambiente de desenvolvimento:

pip install -r requirements-dev.txt
2 Treinar modelo
python scripts/train_pipeline.py
3 Executar API
uvicorn app.main:app --reload
4 Executar monitoramento
python monitoring/drift_dashboard.py
Monitoramento de Predições

Todas as predições realizadas pela API são registradas em:

logs/predictions.json

Esses dados podem ser utilizados para:

auditoria

análise de desempenho

detecção de drift

Possíveis Melhorias Futuras

automação de retraining

deploy em infraestrutura cloud

integração com sistemas educacionais

dashboard em tempo real para monitoramento

orquestração de pipelines com Airflow ou Prefect

Conclusão

Este projeto implementa uma solução completa de Machine Learning em produção, contemplando desde a preparação de dados até o monitoramento contínuo do modelo. A arquitetura foi desenvolvida seguindo boas práticas de engenharia de software e MLOps, garantindo modularidade, reprodutibilidade e facilidade de manutenção.