import pandas as pd

def bin_idade(df):
    df["IDADE_FAIXA"] = pd.cut(
        df["IDADE_ALUNO"],
        bins=[0, 10, 12, 14, 16, 20, 100],
        labels=["<10", "10-12", "12-14", "14-16", "16-20", "20+"]
    )
    return df

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
    df["DEFASADO"] = (df["DEFASAGEM"] > 0).astype(int)
    df["NOTA_MEDIA"] = df[["NOTA_PORT", "NOTA_MAT", "NOTA_ING"]].mean(axis=1)
    df["TEMPO_PM"] = df["ANO_BASE"] - df["ANO_INGRESSO"]
    df["ALTO_INDE"] = (df["INDE"] >= 0.7).astype(int)
    return df

CATEGORICAL_FEATURES = [
    "FASE",
    "TURMA",
    "IDADE_FAIXA"
]

NUMERICAL_FEATURES = [
    "INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN",
    "DEFASAGEM", "NOTA_MEDIA", "TEMPO_PM", "PEDRA_ORDINAL"
]