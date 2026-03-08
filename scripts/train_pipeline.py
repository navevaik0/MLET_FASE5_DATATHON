# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
from pathlib import Path

sys.path.append(os.path.abspath('..'))

# importar funções úteis do módulo de preprocessing
from src.preprocessing import *
from src.feature_engineering import *
from src.train import *
from src.utils import *
from src.evaluate import *
from src.feature_selection import *
from scripts.data_pipeline import main as data_processing

#Parâmetros globais
target_col = "DEFASADO"
vars_fe = ["DEFASADO", "DEFASAGEM", "ANO_BASE", "ALTO_INDE", "INDE"]

# Cria diretórios
model_dir = Path("../app/model")
model_dir.mkdir(exist_ok=True)

artifacts_dir = Path("../artifacts")
artifacts_dir.mkdir(exist_ok=True)

data_dir = Path("../data/processed")
data_dir.mkdir(exist_ok=True)
files = [f for f in data_dir.glob("*.xlsx") if not f.name.startswith("~$")]

data_path = data_dir / "data_processed.parquet"

#Carrega as variáveis finalistas
numeric_cols = load_list(artifacts_dir / "numeric_cols_cat.json")
categorical_cols = load_list(artifacts_dir / "categorical_cols.json")

def main():

    data_processing()

    for file in files:
     df = pd.read_excel(file, engine="openpyxl")

    #Cria uma lista com as variáveis categorizadas
    numeric_cols_cat = [c for c in numeric_cols if c not in [*vars_fe, *[c for c in numeric_cols if c.endswith("_MISSING")]]]

    #Realiza análise de correlação
    corr_matrix, high_corr = feature_correlation_analysis(
        df[[*numeric_cols_cat, target_col]],
        numeric_cols_cat
    )

    #Plota o heatmap da matriz de correlação
    print("Matriz de correlação: \n")
    plot_correlation_matrix(corr_matrix)

    #Tabela de variáveis altamente correlacionadas
    print("Variáveis altamente correlacionadas: \n")
    print(high_corr)

    #Remove as features altamente correlacionadas
    features_remove = remove_features_high_correlation(df, numeric_cols_cat)
    features_numeric_final = [c for c in numeric_cols_cat if c not in features_remove]

    #Verifica alta correlação com a variável alvo
    var_correlation_target= correlation_with_target(df, target_col, features_numeric_final)
    high_corr_target = (
        var_correlation_target
        .loc[var_correlation_target["correlation"] > 0.9, "feature"]
        .tolist()
    )

    features_numeric_final = [c for c in features_numeric_final if c not in high_corr_target]

    print(f"\nFeatures removidas por alta correlação: {features_remove} e {high_corr_target}")
    print(f"\nFeatures finais: {features_numeric_final}")

    categorical_cols_final = [c for c in categorical_cols if c in ['FASE', 'TURMA']]

    print(f"\nColunas categóricas finais: {categorical_cols_final}")
    print(f"\nColunas numéricas finais: {features_numeric_final}")

    # Cria DF do modelo
    df_model = df.copy()

    #Divisão do modelo em treino e teste
    model, X_train, X_test, y_train, y_test = train_model(
        df=df_model,
        numeric_features=features_numeric_final,
        categorical_features=categorical_cols_final,
        target_col=target_col,
        test_year=2024
    )

    metrics_df, _, metrics_dict = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nMétricas do modelo base:")
    print(metrics_df)

    results_df = compare_models(
    df=df_model,
    numeric_features=features_numeric_final,
    categorical_features=categorical_cols_final,
    target_col=target_col,
    test_year=2024
    )

    results_long = results_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        var_name="metric",
        value_name="value"
    )

    print("\nResultados comparativos dos modelos:")
    print(results_long)

    #Treina o modelo campeão para salvar
    final_model, X_train_final, X_test_final, y_train_final, y_test_final = train_champion_model(
        df=df_model,
        numeric_features=features_numeric_final,
        categorical_features=categorical_cols_final,
        target_col=target_col,
        test_year=2024
    )

    # Salva modelo
    save_model(model, model_dir / "model_base.joblib")
    save_model(final_model, model_dir / "model_champion.joblib")

    # Cria datasets
    train_df = pd.concat([X_train_final, y_train_final], axis=1)
    test_df = pd.concat([X_test_final, y_test_final], axis=1)

    # Normaliza tipos
    train_df = train_df.convert_dtypes()
    test_df = test_df.convert_dtypes()

    # Salva parquet
    train_df.to_parquet(
        artifacts_dir / "train_data.parquet",
        engine="pyarrow"
    )

    test_df.to_parquet(
        artifacts_dir / "test_data.parquet",
        engine="pyarrow"
    )

    metrics_df_champion, _, metrics_dict_champion = evaluate_model(
        final_model,
        X_test_final,
        y_test_final
    )

    update_metadata(
        metadata_path= artifacts_dir / "model_metadata.json",
        model_name="baseline_logistic",
        version="v1",
        metrics=metrics_dict,
        features= list(set(features_numeric_final + categorical_cols_final)),
        champion=False
    )

    update_metadata(
        metadata_path= artifacts_dir / "model_metadata.json",
        model_name="champion_xgboost",
        version="v2",
        metrics=metrics_dict_champion,
        features= list(set(features_numeric_final + categorical_cols_final)),
        champion=True
    )

if __name__ == "__main__":
    main()