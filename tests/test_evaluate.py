import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from src.evaluate import *
from src.train import *


def test_compute_metrics_retorna_dict():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0

def test_metrics_to_dataframe():
    metrics = {
        "accuracy": 0.8,
        "precision": 0.7
    }

    df = metrics_to_dataframe(metrics)

    assert "metric" in df.columns
    assert "value" in df.columns
    assert len(df) == 2

def test_classification_report_df_retorna_dataframe():
    import numpy as np
    import pandas as pd

    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    df = classification_report_df(y_true, y_pred)

    assert "label" in df.columns
    assert len(df) > 0

def test_evaluate_model_funciona():
    df = pd.DataFrame({
        "INDE": [1, 2, 3, 4],
        "DEFASAGEM": [0, 1, 0, 1],
        "FASE": ["A", "B", "A", "B"],
        "TURMA": ["X", "X", "Y", "Y"],
        "IDADE_FAIXA": ["10-12", "10-12", "13-15", "13-15"],
        "TARGET": [0, 1, 0, 1],
        "ANO_BASE": [2022, 2022, 2023, 2023]
    })

    model = build_model(
        numeric_features=["INDE", "DEFASAGEM"],
        categorical_features=["FASE", "TURMA", "IDADE_FAIXA"]
    )

    model.fit(df.drop(columns=["TARGET"]), df["TARGET"])

    metrics_df, report_df, metrics_dict = evaluate_model(
        model,
        df.drop(columns=["TARGET"]),
        df["TARGET"]
    )

    assert not metrics_df.empty
    assert not report_df.empty
    assert isinstance(metrics_dict, dict)

def test_plot_functions_retornam_figure():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])

    fig1 = plot_confusion_matrix(y_true, y_pred)
    fig2 = plot_roc_curve(y_true, y_proba)
    fig3 = plot_precision_recall_curve(y_true, y_proba)

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None

from src.train import build_model
from src.evaluate import evaluate_model


def test_evaluate_model_com_plot():
    df = pd.DataFrame({
        "INDE": [1, 2, 3, 4],
        "DEFASAGEM": [0, 1, 0, 1],
        "FASE": ["A", "B", "A", "B"],
        "TURMA": ["X", "X", "Y", "Y"],
        "IDADE_FAIXA": ["10-12", "10-12", "13-15", "13-15"],
        "TARGET": [0, 1, 0, 1],
        "ANO_BASE": [2022, 2022, 2023, 2023]
    })

    model = build_model(
        numeric_features=["INDE", "DEFASAGEM"],
        categorical_features=["FASE", "TURMA", "IDADE_FAIXA"]
    )

    model.fit(df.drop(columns=["TARGET"]), df["TARGET"])

    metrics_df, report_df, metrics_dict = evaluate_model(
        model,
        df.drop(columns=["TARGET"]),
        df["TARGET"],
        plot=True
    )

    assert not metrics_df.empty
    assert not report_df.empty