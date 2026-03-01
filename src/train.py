import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

TARGET_COL = "ALTO_INDE"

NUMERICAL_FEATURES = [
    "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN",
    "DEFASADO", "NOTA_MEDIA", "TEMPO_PM"
]

CATEGORICAL_FEATURES = [
    "FASE",
    "TURMA",
    "IDADE_FAIXA"
]

def build_preprocessor():
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
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor

def build_model():
    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=None
            ))
        ]
    )

    return model

def temporal_train_test_split(
    df: pd.DataFrame,
    test_year: int
):
    train_df = df[df["ANO_BASE"] < test_year]
    test_df = df[df["ANO_BASE"] == test_year]

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test

def train(
    df: pd.DataFrame,
    test_year: int = 2024
):
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df, test_year
    )

    model = build_model()
    model.fit(X_train, y_train)

    return model, X_test, y_test
