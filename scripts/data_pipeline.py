# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
from pathlib import Path

sys.path.append(os.path.abspath('..'))

# importar funções úteis do módulo de preprocessing
from src.preprocessing import *
from src.feature_engineering import *
from src.train import *
from src.utils import *
from src.evaluate import *
from src.feature_selection import *

#Parâmetros globais
target_col = "DEFASADO"
vars_fe = ["DEFASADO", "DEFASAGEM", "ANO_BASE", "ALTO_INDE", "INDE"]
file_path = r'C:\Users\erick\Documents\FIAP\TC\FASE5_DATATHON\data\raw\base_PEDE.xlsx'

# Cria diretórios
model_dir = Path("../app/model")
model_dir.mkdir(exist_ok=True)

artifacts_dir = Path("../artifacts")
artifacts_dir.mkdir(exist_ok=True)

data_dir = Path("../data/processed")
data_dir.mkdir(exist_ok=True)


def main():

    # Carregar os dados de cada ano
    df_2022 = pd.read_excel(file_path, sheet_name='PEDE2022')
    df_2023 = pd.read_excel(file_path, sheet_name='PEDE2023')
    df_2024 = pd.read_excel(file_path, sheet_name='PEDE2024')

    # Padronizar colunas
    df22 = padronizar_colunas(df_2022, column_mapping, year=2022, return_summary=False)
    df23 = padronizar_colunas(df_2023, column_mapping, year=2023, return_summary=False)
    df24 = padronizar_colunas(df_2024, column_mapping, year=2024, return_summary=False)

    # Concatenar dataframes
    df_concat = pd.concat([df22, df23, df24], ignore_index=True)

    # Aplicar correção de idade e enforcer de schema
    df_fix_idade = fix_idade(df_concat)
    df_schema, schema_errors = enforce_schema(df_fix_idade, schema= EXPECTED_SCHEMA)

    #Separa as colunas numéricas e categóricas para as próximas etapas
    numeric_cols, categorical_cols = split_feature_types(
        df_schema,
        target_col=target_col
    )

    #Tratamento de missings
    df_pede, missing_summary = handle_missing_values(
        df_schema,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        add_missing_flags=True
    )

    #Feature engineering
    #Aplica as transformações iniciais
    df_fe = (
        df_pede
        .pipe(encode_pedra)
        .pipe(create_basic_features)
    )

    #Separa novamente as colunas numéricas e categóricas para as próximas etapas
    numeric_cols, categorical_cols = split_feature_types(
        df_fe,
        target_col=target_col
    )

    #Aplica binning
    df_fe, binning_dict = create_numeric_binning(
        df_fe,
        numeric_cols,
        bins=5,
        exceptions=[*vars_fe, *[c for c in numeric_cols if c.endswith("_MISSING")]]
    )

    #Salva dicionários
    save_dict(binning_dict, artifacts_dir / "binning_dict.json")

    #Garante que as variáveis para separação do modelo estão no dataframe
    assert target_col in df_fe.columns, f"Coluna alvo '{target_col}' não encontrada no dataframe."
    assert "ANO_BASE" in df_fe.columns, "Coluna ANO_BASE não encontrada no dataframe."

    #Cria uma lista com as variáveis categorizadas
    numeric_cols_cat = ["CAT_" + c for c in numeric_cols if c not in [*vars_fe, *[c for c in numeric_cols if c.endswith("_MISSING")]]]

    save_list(numeric_cols, artifacts_dir /"numeric_cols.json")
    save_list(numeric_cols_cat, artifacts_dir /"numeric_cols_cat.json")
    save_list(categorical_cols, artifacts_dir /"categorical_cols.json")
    
    #Salva a base tratada
    df_fe.to_excel(data_dir/'data_processed.xlsx', index=False)
    
if __name__ == "__main__":
    main()