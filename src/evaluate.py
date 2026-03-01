import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_metrics(model, X_test, y_test):
    """
    Retorna métricas principais em formato tabular.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    metrics_df = (
        pd.DataFrame(metrics, index=["value"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    return metrics_df

def classification_report_df(model, X_test, y_test):
    """
    Retorna classification report em DataFrame.
    """
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
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

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True)
    plt.show()    

def plot_proba_distribution(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(6, 4))
    plt.hist(y_proba[y_test == 0], bins=30, alpha=0.6, label="Classe 0")
    plt.hist(y_proba[y_test == 1], bins=30, alpha=0.6, label="Classe 1")

    plt.xlabel("Probabilidade predita")
    plt.ylabel("Frequência")
    plt.title("Distribuição das Probabilidades")
    plt.legend()
    plt.show()

def plot_precision_recall_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()