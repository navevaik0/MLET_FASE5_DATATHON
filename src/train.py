from typing import List, Tuple
import pandas as pd
from src.evaluate import evaluate_model

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Cria o pré-processador (scaling + OHE).
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor


def build_model(
    numeric_features,
    categorical_features,
    model_type: str = "logistic"
):
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    if model_type == "logistic":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    elif model_type == "xgboost":
        clf = XGBClassifier(
            n_estimators=200,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )

    else:
        raise ValueError("Modelo não suportado.")

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

    return model


def temporal_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_year: int,
    year_col: str = "ANO_BASE"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide treino e teste baseado em ano.
    """

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' não encontrada no DataFrame.")

    if year_col not in df.columns:
        raise ValueError(f"Coluna de ano '{year_col}' não encontrada.")

    train_df = df[df[year_col] < test_year]
    test_df = df[df[year_col] == test_year]

    if train_df.empty:
        raise ValueError("Conjunto de treino vazio.")

    if test_df.empty:
        raise ValueError("Conjunto de teste vazio.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test


def train_model(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_col: str,
    test_year: int,
    year_col: str = "ANO_BASE"
):
    """
    Executa split temporal + treino.
    """

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df=df,
        target_col=target_col,
        test_year=test_year,
        year_col=year_col
    )

    model = build_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def compare_models(
    df,
    numeric_features,
    categorical_features,
    target_col,
    test_year,
    models=("logistic", "random_forest", "xgboost")
):
    from src.train import build_model, temporal_train_test_split
    from src.evaluate import evaluate_model

    results = []

    # split uma única vez (importante)
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df=df,
        target_col=target_col,
        test_year=test_year
    )

    for model_name in models:

        model = build_model(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            model_type=model_name
        )

        model.fit(X_train, y_train)

        _, _, metrics_dict = evaluate_model(
            model,
            X_test,
            y_test,
            plot=False
        )

        metrics_dict["model"] = model_name
        results.append(metrics_dict)

    return pd.DataFrame(results)