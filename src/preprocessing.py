import pandas as pd
from typing import List, Tuple, Dict

dicionario_dados = {
   "NOME":"Nome do aluno (dados anonimizados)",
   "INSTITUICAO_ENSINO_ALUNO":"Instituição de ensino do aluno",
   "IDADE_ALUNO":"Idade do aluno",
   #"ANOS_NA_PM":"Tempo (em anos) que o aluno está na Passos Mágicos",
   "FASE":"Fase (nível de aprendizado) e turma do aluno",
   #"PONTO_VIRADA":"Indica se o aluno atingiu o Ponto de Virada  (booleano)",
   "INDE": "Índice de Desenvolvimento Educacional , calculado pela ponderação dos indicadores IAN, IDA, IEG, IAA, IPS, IPP e IPV",
   #"INDE_CONCEITO":"Conceito associado ao valor do INDE",
   "PEDRA":"Classificação do aluno com base no INDE (Quartzo, Ágata, Ametista, Topázio)",
   "IAA":"Indicador de Autoavaliação - média das notas",
   "IEG":"Indicador de Engajamento - média das notas",
   "IPS":"Indicador Psicossocial - média das notas",
   "IDA":"Indicador de Aprendizagem - média das notas",
   "IPP":"Indicador Psicopedagógico - média das notas",
   "IPV":"Indicador de Ponto de Virada - média das notas",
   "IAN":"Indicador de Adequação ao Nível - média das notas",
   "DESTAQUE_IEG":"Observações dos avaliadores sobre o Engajamento",
   "DESTAQUE_IDA":"Observações dos avaliadores sobre a Aprendizagem",
   "DESTAQUE_IPV":"Observações dos avaliadores sobre o Ponto de Virada",
   "TURMA":"Turma do aluno",
   #"SINALIZADOR_INGRESSANTE":"Indica se o aluno é ingressante ou veterano",
   "REC_PSICO":"Recomendação da equipe de psicologia",
   "REC_AVAL_1":"Recomendação da equipe de avaliação 1",
   "REC_AVAL_2":"Recomendação da equipe de avaliação 2",
   "REC_AVAL_3":"Recomendação da equipe de avaliação 3",
   "REC_AVAL_4":"Recomendação da equipe de avaliação 4",
   "NIVEL_IDEAL":"Nível (fase) ideal do aluno",
   "DEFASAGEM":"Nível de defasagem do aluno",
   "ANO_INGRESSO":"Ano de ingresso do aluno na Passos Mágicos",
   #"BOLSISTA":"Indica se o aluno é bolsista",
   "CG":"Classificação geral (ranking) do aluno",
   "CF":"Classificação do aluno na fase",
   "CT":"Classificação do aluno na turma",
   "NOTA_PORT":"Média das notas de Português",
   "NOTA_MAT":"Média das notas de Matemática",
   "NOTA_ING":"Média das notas de Inglês",
   "QTDE_AVAL":"Quantidade de avaliações realizadas",
}

EXPECTED_SCHEMA = {
    "NOME": "string",
    "INSTITUICAO_ENSINO_ALUNO": "string",
    "IDADE_ALUNO": "int",
    "FASE": "string",
    "INDE": "float",
    "PEDRA": "string",
    "IAA": "float",
    "IEG": "float",
    "IPS": "float",
    "IDA": "float",
    "IPP": "float",
    "IPV": "float",
    "IAN": "float",
    "TURMA": "string",
    "REC_PSICO": "string",
    "REC_AVAL_1": "string",
    "REC_AVAL_2": "string",
    "REC_AVAL_3": "string",
    "REC_AVAL_4": "string",
    "NIVEL_IDEAL": "string",
    "DEFASAGEM": "float",
    "ANO_INGRESSO": "int",
    "CG": "float",
    "CF": "float",
    "CT": "float",
    "NOTA_PORT": "float",
    "NOTA_MAT": "float",
    "NOTA_ING": "float",
    "QTDE_AVAL": "int",
    "ANO_BASE": "int",
}

column_mapping = {
    2022: {
        "RA": "RA",
        "Nome": "NOME",
        "Instituição de ensino": "INSTITUICAO_ENSINO_ALUNO",
        "Idade 22": "IDADE_ALUNO",
        "Fase": "FASE",
        "INDE 22": "INDE",
        "Pedra 22": "PEDRA",
        "IAA": "IAA",
        "IEG": "IEG",
        "IPS": "IPS",
        "IDA": "IDA",
        "IPP": "IPP",
        "IPV": "IPV",
        "IAN": "IAN",
        "Destaque IEG": "DESTAQUE_IEG",
        "Destaque IDA": "DESTAQUE_IDA",
        "Destaque IPV": "DESTAQUE_IPV",
        "Turma": "TURMA",
        "Rec Psicologia": "REC_PSICO",
        "Rec Av1": "REC_AVAL_1",
        "Rec Av2": "REC_AVAL_2",
        "Rec Av3": "REC_AVAL_3",
        "Rec Av4": "REC_AVAL_4",
        "Fase ideal": "NIVEL_IDEAL",
        "Defas": "DEFASAGEM",
        "Ano ingresso": "ANO_INGRESSO",
        "Cg": "CG",
        "Cf": "CF",
        "Ct": "CT",
        "Portug": "NOTA_PORT",
        "Matem": "NOTA_MAT",
        "Inglês": "NOTA_ING",
        "Nº Av": "QTDE_AVAL"
    },

    2023: {
        "RA": "RA",
        "Nome Anonimizado": "NOME",
        "Instituição de ensino": "INSTITUICAO_ENSINO_ALUNO",
        "Idade": "IDADE_ALUNO",
        "Fase": "FASE",
        "INDE 2023": "INDE",
        "Pedra 2023": "PEDRA",
        "IAA": "IAA",
        "IEG": "IEG",
        "IPS": "IPS",
        "IDA": "IDA",
        "IPP": "IPP",
        "IPV": "IPV",
        "IAN": "IAN",
        "Destaque IEG": "DESTAQUE_IEG",
        "Destaque IDA": "DESTAQUE_IDA",
        "Destaque IPV": "DESTAQUE_IPV",
        "Turma": "TURMA",
        "Rec Psicologia": "REC_PSICO",
        "Rec Av1": "REC_AVAL_1",
        "Rec Av2": "REC_AVAL_2",
        "Rec Av3": "REC_AVAL_3",
        "Rec Av4": "REC_AVAL_4",
        "Fase Ideal": "NIVEL_IDEAL",
        "Defasagem": "DEFASAGEM",
        "Ano ingresso": "ANO_INGRESSO",
        "Cg": "CG",
        "Cf": "CF",
        "Ct": "CT",
        "Por": "NOTA_PORT",
        "Mat": "NOTA_MAT",
        "Ing": "NOTA_ING",
        "Nº Av": "QTDE_AVAL"
    },

    2024: {
        "RA": "RA",
        "Nome Anonimizado": "NOME",
        "Instituição de ensino": "INSTITUICAO_ENSINO_ALUNO",
        "Idade": "IDADE_ALUNO",
        "Fase": "FASE",
        "INDE 2024": "INDE",
        "Pedra 2024": "PEDRA",
        "IAA": "IAA",
        "IEG": "IEG",
        "IPS": "IPS",
        "IDA": "IDA",
        "IPP": "IPP",
        "IPV": "IPV",
        "IAN": "IAN",
        "Destaque IEG": "DESTAQUE_IEG",
        "Destaque IDA": "DESTAQUE_IDA",
        "Destaque IPV": "DESTAQUE_IPV",
        "Turma": "TURMA",
        "Rec Psicologia": "REC_PSICO",
        "Rec Av1": "REC_AVAL_1",
        "Rec Av2": "REC_AVAL_2",
        "Rec Av3": "REC_AVAL_3",
        "Rec Av4": "REC_AVAL_4",
        "Fase Ideal": "NIVEL_IDEAL",
        "Defasagem": "DEFASAGEM",
        "Ano ingresso": "ANO_INGRESSO",
        "Cg": "CG",
        "Cf": "CF",
        "Ct": "CT",
        "Por": "NOTA_PORT",
        "Mat": "NOTA_MAT",
        "Ing": "NOTA_ING",
        "Nº Av": "QTDE_AVAL"
    }
}

NUMERIC_COLS = [
    "INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN",
    "DEFASAGEM", "NOTA_PORT", "NOTA_MAT", "NOTA_ING",
    "ANO_INGRESSO", "QTDE_AVAL"
]

CATEGORICAL_COLS = [
    "FASE", "TURMA", "PEDRA", "NIVEL_IDEAL"
]

#### FUNÇÕES
def padronizar_colunas(
    df: pd.DataFrame,
    column_map: dict,
    year: int | None = None,
    return_summary: bool = True
):
    """
    Renomeia e filtra as colunas de um DataFrame usando um mapeamento
    explicitamente fornecido.

    Se `column_map` contiver chaves numéricas (anos), `year` determina
    qual sub-mapa usar. Caso contrário, assume-se que `column_map`
    já é um mapa direto {orig: novo}.

    Apenas as colunas renomeadas são mantidas no resultado; todas as
    outras são descartadas.

    Retorna o DataFrame padronizado e, opcionalmente, um resumo com:
      * renomeadas: colunas originais → nova
      * excluidas: colunas originais que foram descartadas
    """

    df = df.copy()
    cols_originais = list(df.columns)

    # selecionar o mapa correto de acordo com o ano
    if year is not None and year in column_map:
        mapa = column_map[year]
    else:
        mapa = column_map

    # construir resumo
    resumo = {'renomeadas': {}, 'excluidas': []}

    # filtrar entradas cujo original exista no DataFrame
    valid_map = {orig: novo for orig, novo in mapa.items() if orig in cols_originais}

    # renomear
    df = df.rename(columns=valid_map)

    # registrar renomeações não triviais
    for orig, novo in valid_map.items():
        if orig != novo:
            resumo['renomeadas'][orig] = novo

    # manter apenas as colunas resultantes do mapeamento
    colunas_resultantes = list(valid_map.values())
    df_padronizado = df[[c for c in df.columns if c in colunas_resultantes]].copy()

    # se ano for especificado, adicionar coluna de origem
    if year is not None:
        df_padronizado["ANO_BASE"] = int(year)

    # colunas originais que não apareceram no mapeamento são excluídas
    resumo['excluidas'] = [c for c in cols_originais if c not in valid_map]

    if return_summary:
        return df_padronizado, resumo
    else:
        return df_padronizado

def enforce_schema(df: pd.DataFrame, schema: dict, strict: bool = False):
    df = df.copy()
    errors = []

    for col, dtype in schema.items():
        if col not in df.columns:
            msg = f"Coluna ausente: {col}"
            if strict:
                raise ValueError(msg)
            errors.append(msg)
            continue

        try:
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "string":
                df[col] = df[col].astype("string")
        except Exception as e:
            errors.append(f"Erro ao converter {col}: {e}")

    return df, errors

def fix_idade(df: pd.DataFrame):
    if pd.api.types.is_datetime64_any_dtype(df["IDADE_ALUNO"]):
        df["IDADE_ALUNO"] = (df["IDADE_ALUNO"] - pd.Timestamp("1900-01-01")).dt.days
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    add_missing_flags: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Trata valores nulos em colunas numéricas e categóricas.

    - Numéricas: preenche com a mediana
    - Categóricas: preenche com 'DESCONHECIDO'
    - Opcional: cria flags *_MISSING para colunas com nulos

    Retorna:
      - DataFrame tratado
      - Dicionário com resumo das imputações
    """

    df = df.copy()
    summary = {
        "numeric_imputed": {},
        "categorical_imputed": {},
        "missing_flags_created": []
    }

    # -------------------------
    # Numéricas
    # -------------------------
    for col in numeric_cols:
        if col not in df.columns:
            continue

        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

            summary["numeric_imputed"][col] = {
                "missing_count": int(n_missing),
                "imputed_with": float(median_value)
            }

            if add_missing_flags:
                flag_col = f"{col}_MISSING"
                df[flag_col] = (df[col].isna()).astype(int)
                summary["missing_flags_created"].append(flag_col)

    # -------------------------
    # Categóricas
    # -------------------------
    for col in categorical_cols:
        if col not in df.columns:
            continue

        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna("DESCONHECIDO")

            summary["categorical_imputed"][col] = {
                "missing_count": int(n_missing),
                "imputed_with": "DESCONHECIDO"
            }

            if add_missing_flags:
                flag_col = f"{col}_MISSING"
                df[flag_col] = (df[col] == "DESCONHECIDO").astype(int)
                summary["missing_flags_created"].append(flag_col)
    
    # Flags informativas
    for col in ["CF", "CT"]:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(-1)

    return df, summary