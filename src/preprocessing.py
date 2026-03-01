import pandas as pd
import re

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
        df_padronizado["ANO_BASE"] = str(year)

    # colunas originais que não apareceram no mapeamento são excluídas
    resumo['excluidas'] = [c for c in cols_originais if c not in valid_map]

    if return_summary:
        return df_padronizado, resumo
    else:
        return df_padronizado

