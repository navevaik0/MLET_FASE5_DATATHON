import pandas as pd
from src.feature_engineering import *

def test_encode_pedra_cria_coluna_ordinal():
    df = pd.DataFrame({
        "PEDRA": ["Quartzo", "Ágata"]
    })

    df_encoded = encode_pedra(df)

    assert "PEDRA_ORDINAL" in df_encoded.columns
    assert df_encoded["PEDRA_ORDINAL"].notna().all()

def test_bin_cria_coluna():
    df = pd.DataFrame({
        "IDADE_ALUNO": [10, 15, 18]
    })

    df_out, binning_dict = create_numeric_binning(df, numeric_cols=["IDADE_ALUNO"])

    assert "CAT_IDADE_ALUNO" in df_out.columns
    assert "IDADE_ALUNO" not in df_out.columns
    assert "IDADE_ALUNO" in binning_dict

def test_bin_aplica_coluna():
    df = pd.DataFrame({
        "IDADE_ALUNO": [10, 15, 18]
    })

    binning_dict = {
        "IDADE_ALUNO": [10, 12, 14, 16, 18, 20]
    }

    df_out = apply_numeric_binning(df, binning_dict)

    assert "CAT_IDADE_ALUNO" in df_out.columns
    assert "IDADE_ALUNO" not in df_out.columns


def test_binning_valores_iguais():

    df = pd.DataFrame({
        "IDADE_ALUNO": [10,10,10,10]
    })

    df_out, binning_dict = create_numeric_binning(
        df,
        numeric_cols=["IDADE_ALUNO"],
        bins=5
    )

    assert "CAT_IDADE_ALUNO" in df_out.columns

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