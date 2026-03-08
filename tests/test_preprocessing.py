import pandas as pd
import pytest
from src.preprocessing import *
from src.train import *

def test_padronizar_colunas_renomeia_corretamente():
    df = pd.DataFrame({
        "Nome": ["A", "B"],
        "Idade 22": [10, 12]
    })

    mapping = {
        2022: {
            "Nome": "NOME",
            "Idade 22": "IDADE_ALUNO"
        }
    }

    df_pad = padronizar_colunas(df, mapping, year=2022, return_summary=False)

    assert "NOME" in df_pad.columns
    assert "IDADE_ALUNO" in df_pad.columns
    assert "Nome" not in df_pad.columns


def test_handle_missing_values_numeric():
    df = pd.DataFrame({
        "INDE": [1.0, None, 3.0]
    })

    df_clean, _ = handle_missing_values(
        df,
        numeric_cols=["INDE"],
        categorical_cols=[]
    )

    assert df_clean["INDE"].isna().sum() == 0

def test_split_erro_quando_target_inexistente():
    df = pd.DataFrame({
        "A": [1,2],
        "ANO_BASE": [2022, 2023]
    })

    with pytest.raises(ValueError):
        temporal_train_test_split(
            df,
            target_col="TARGET",
            test_year=2023
        )


def test_enforce_schema_converte_int():
    schema = {"IDADE": "int"}

    df = pd.DataFrame({
        "IDADE": ["10", "20"]
    })

    df_out, errors = enforce_schema(df, schema)

    assert errors == []
    assert str(df_out["IDADE"].dtype) == "Int64"
    assert df_out["IDADE"].iloc[0] == 10

def test_enforce_schema_converte_float():
    schema = {"INDE": "float"}

    df = pd.DataFrame({
        "INDE": ["7.5", "8.0"]
    })

    df_out, errors = enforce_schema(df, schema)

    assert errors == []
    assert df_out["INDE"].dtype == "float64"


def test_enforce_schema_converte_string():
    schema = {"NOME": "string"}

    df = pd.DataFrame({
        "NOME": [123, 456]
    })

    df_out, errors = enforce_schema(df, schema)

    assert errors == []
    assert str(df_out["NOME"].dtype) == "string"   

def test_enforce_schema_coluna_ausente_nao_strict():
    schema = {"INDE": "float"}

    df = pd.DataFrame({
        "IDADE": [10, 20]
    })

    df_out, errors = enforce_schema(df, schema)

    assert len(errors) == 1
    assert "Coluna ausente: INDE" in errors[0]

def test_enforce_schema_coluna_ausente_nao_strict():
    schema = {"INDE": "float"}

    df = pd.DataFrame({
        "IDADE": [10, 20]
    })

    df_out, errors = enforce_schema(df, schema)

    assert len(errors) == 1
    assert "Coluna ausente: INDE" in errors[0]

def test_enforce_schema_int_invalido_vira_na():
    schema = {"IDADE": "int"}

    df = pd.DataFrame({
        "IDADE": ["10", "abc"]
    })

    df_out, errors = enforce_schema(df, schema)

    assert errors == []
    assert pd.isna(df_out["IDADE"].iloc[1])

def test_split_feature_types():

    df = pd.DataFrame({
        "idade": [10, 12, 14],
        "nota": [7.5, 8.0, 9.0],
        "turma": ["A", "B", "A"],
        "target": [0, 1, 1]
    })

    numeric_cols, categorical_cols = split_feature_types(
        df,
        target_col="target"
    )

    assert "idade" in numeric_cols
    assert "nota" in numeric_cols
    assert "turma" in categorical_cols