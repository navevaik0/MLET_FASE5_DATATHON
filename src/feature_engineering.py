import pandas as pd
import numpy as np

PEDRA_MAP = {
    "Quartzo": 0,
    "Ágata": 1,
    "Agata": 1,
    "Ametista": 2,
    "Topázio": 3,
    "INCLUIR": 4,
    "DESCONHECIDO": 4, 
}

def encode_pedra(df):
    df["PEDRA_ORDINAL"] = df["PEDRA"].map(PEDRA_MAP)
    return df

def create_basic_features(df):
    df["DEFASADO"] = (df["DEFASAGEM"] < 0).astype(int)
    df["NOTA_MEDIA"] = df[["NOTA_PORT", "NOTA_MAT", "NOTA_ING"]].mean(axis=1)
    df["TEMPO_PM"] = df["ANO_BASE"] - df["ANO_INGRESSO"]
    df["ALTO_INDE"] = (df["INDE"] >= 7).astype(int)
    return df

def create_numeric_binning(
    df: pd.DataFrame,
    numeric_cols: list,
    bins: int = 5,
    exceptions: list = [],
):
    """
    Cria bins numéricos ordinalizados (1..bins).
    Remove variável original e cria CAT_variavel.
    """

    df_out = df.copy()
    binning_dict = {}
    numeric_cols_final = [col for col in numeric_cols if col not in exceptions]

    for col in numeric_cols_final:

        # calcula edges
        _, edges = pd.qcut(
            df[col],
            q=bins,
            retbins=True,
            duplicates="drop"
        )

        binning_dict[col] = edges.tolist()

        # aplica binning ordinal
        df_out[f"CAT_{col}"] = np.digitize(
            df[col],
            bins=edges[1:-1],
            right=True
        ) + 1

        # remove variável original
        df_out.drop(columns=[col], inplace=True)

    return df_out, binning_dict

def apply_numeric_binning(
    df: pd.DataFrame,
    binning_dict: dict
):
    """
    Aplica binning usando limites previamente calculados.
    """

    df_out = df.copy()

    for col, edges in binning_dict.items():

        df_out[f"CAT_{col}"] = np.digitize(
            df[col],
            bins=edges[1:-1],
            right=True
        ) + 1

        df_out.drop(columns=[col], inplace=True)

    return df_out