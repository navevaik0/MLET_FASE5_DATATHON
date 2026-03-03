import pandas as pd
from src.feature_engineering import *

def test_encode_pedra_cria_coluna_ordinal():
    df = pd.DataFrame({
        "PEDRA": ["Quartzo", "Ágata"]
    })

    df_encoded = encode_pedra(df)

    assert "PEDRA_ORDINAL" in df_encoded.columns
    assert df_encoded["PEDRA_ORDINAL"].notna().all()

def test_bin_idade_cria_coluna():
    df = pd.DataFrame({
        "IDADE_ALUNO": [10, 15, 18]
    })

    df_out = bin_idade(df)

    assert "IDADE_FAIXA" in df_out.columns

def test_create_basic_feature_cria_coluna():
    df = pd.DataFrame({
        "DEFASAGEM": [-1, 0, 3],
        "NOTA_PORT": [8, 9, 8],
        "NOTA_MAT": [6, 5, 4],
        "NOTA_ING": [7, 7, 8],
        "ANO_INGRESSO": [2020, 2015, 2028],
        "ANO_BASE": [2022, 2023, 2023],
        "INDE": [7, 8, 5],
    })

    df_out = create_basic_features(df)

    assert "DEFASADO" in df_out.columns    