import pandas as pd
from src.train import *

def test_temporal_split():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "TARGET": [0, 1, 0, 1],
        "ANO_BASE": [2022, 2023, 2024, 2024]
    })

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df,
        target_col="TARGET",
        test_year=2024
    )

    assert all(X_train["ANO_BASE"] < 2024)
    assert all(X_test["ANO_BASE"] == 2024)

def test_model_treina():
    df = pd.DataFrame({
        "INDE": [1, 2, 3, 4],
        "DEFASAGEM": [0, 1, 0, 1],
        "FASE": ["A", "B", "A", "B"],
        "TURMA": ["X", "X", "Y", "Y"],
        "IDADE_FAIXA": ["10-12", "10-12", "13-15", "13-15"],
        "TARGET": [0, 1, 0, 1],
        "ANO_BASE": [2022, 2022, 2023, 2023]
    })

    model, _, _, _, _ = train_model(
        df=df,
        numeric_features=["INDE", "DEFASAGEM"],
        categorical_features=["FASE", "TURMA", "IDADE_FAIXA"],
        target_col="TARGET",
        test_year=2023
    )

    assert model is not None