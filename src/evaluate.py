from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas principais de classificação binária.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

def metrics_to_dataframe(metrics: Dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame(metrics, index=["value"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

def classification_report_df(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> pd.DataFrame:

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return (
        pd.DataFrame(report)
        .T
        .reset_index()
        .rename(columns={"index": "label"})
    )


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax
    )

    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")

    return fig

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC")
    ax.legend()
    ax.grid(True)

    return fig

def plot_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True)

    return fig

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    metrics_df = metrics_to_dataframe(metrics)
    report_df = classification_report_df(y_test, y_pred)

    if plot:
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(y_test, y_proba)
        plot_precision_recall_curve(y_test, y_proba)

    return metrics_df, report_df, metrics