import pandas as pd  
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_matrix(
    df: pd.DataFrame,
    numeric_features: list
):
    """
    Retorna matriz de correlação entre variáveis numéricas.
    """

    corr_matrix = df[numeric_features].corr()

    return corr_matrix

def plot_correlation_matrix(
    corr_matrix,
    figsize=(10,8)
):

    plt.figure(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Matriz de Correlação entre Variáveis")
    plt.show()



def detect_high_correlation(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8
):
    """
    Detecta pares de variáveis altamente correlacionadas.
    """

    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i):

            corr_value = corr_matrix.iloc[i, j]

            if abs(corr_value) >= threshold:

                feature_1 = corr_matrix.columns[i]
                feature_2 = corr_matrix.columns[j]

                high_corr_pairs.append({
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": corr_value
                })

    return pd.DataFrame(high_corr_pairs)


def feature_correlation_analysis(
    df: pd.DataFrame,
    numeric_features: list,
    threshold: float = 0.8
):

    corr_matrix = correlation_matrix(df, numeric_features)

    high_corr = detect_high_correlation(
        corr_matrix,
        threshold
    )

    return corr_matrix, high_corr


def remove_features_high_correlation(
    df: pd.DataFrame,
    numeric_features: list,
    threshold: float = 0.8
):

    corr_matrix = correlation_matrix(df, numeric_features)

    high_corr = detect_high_correlation(
        corr_matrix,
        threshold
    )

    features_to_remove = set(high_corr.get('feature_2', []))

    return features_to_remove


def correlation_with_target(
    df: pd.DataFrame,
    target_col: str,
    numeric_features: list
) -> pd.DataFrame:
    """
    Calcula correlação das variáveis numéricas com a target.
    """

    corr_series = df[numeric_features + [target_col]].corr()[target_col]

    corr_df = (
        corr_series
        .drop(target_col)
        .abs()
        .sort_values(ascending=False)
        .reset_index()
    )

    corr_df.columns = ["feature", "correlation"]

    return corr_df


def categorical_target_concentration(
    df: pd.DataFrame,
    target_col: str,
    categorical_features: list
):
    """
    Calcula concentração da target em variáveis categóricas.
    """

    results = {}

    for col in categorical_features:

        concentration = (
            df.groupby(col)[target_col]
            .mean()
            .sort_values(ascending=False)
        )

        results[col] = concentration

    return results

def plot_target_by_bin_grid(
    df,
    target_col,
    bin_prefix="CAT_"
):

    bin_cols = [c for c in df.columns if c.startswith(bin_prefix)]

    n = len(bin_cols)

    fig, axes = plt.subplots(
        nrows=(n//3)+1,
        ncols=3,
        figsize=(14,10)
    )

    axes = axes.flatten()

    for i, col in enumerate(bin_cols):

        summary = (
            df.groupby(col)[target_col]
            .mean()
            .reset_index()
        )

        sns.lineplot(
            data=summary,
            x=col,
            y=target_col,
            marker="o",
            ax=axes[i]
        )

        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()