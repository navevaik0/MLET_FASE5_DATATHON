import pandas as pd
from src.feature_selection import *

def test_correlation_matrix():

    df = pd.DataFrame({
        "a": [1,2,3,4],
        "b": [2,4,6,8],
        "c": [4,3,2,1]
    })

    corr = correlation_matrix(df, ["a","b","c"])

    assert corr.shape == (3,3)
    assert corr.loc["a","b"] > 0.9

def test_detect_high_correlation():

    df = pd.DataFrame({
        "a": [1,2,3,4],
        "b": [2,4,6,8],  # altamente correlacionada
        "c": [1,0,1,0]
    })

    corr = df.corr()

    result = detect_high_correlation(
        corr,
        threshold=0.9
    )

    assert not result.empty
    assert "feature_1" in result.columns
    assert "feature_2" in result.columns

def test_feature_correlation_analysis():

    df = pd.DataFrame({
        "a": [1,2,3,4],
        "b": [2,4,6,8],
        "c": [1,0,1,0]
    })

    corr_matrix, high_corr = feature_correlation_analysis(
        df,
        numeric_features=["a","b","c"],
        threshold=0.9
    )

    assert isinstance(corr_matrix, pd.DataFrame)
    assert isinstance(high_corr, pd.DataFrame)

def test_correlation_with_target():

    df = pd.DataFrame({
        "var1": [1,2,3,4],
        "var2": [4,3,2,1],
        "target": [0,0,1,1]
    })

    corr_df = correlation_with_target(
        df,
        target_col="target",
        numeric_features=["var1","var2"]
    )

    assert "feature" in corr_df.columns
    assert "correlation" in corr_df.columns
    assert len(corr_df) == 2

def test_categorical_target_concentration():

    df = pd.DataFrame({
        "turma": ["A","A","B","B"],
        "target": [0,1,1,1]
    })

    result = categorical_target_concentration(
        df,
        target_col="target",
        categorical_features=["turma"]
    )

    assert "turma" in result
    assert isinstance(result["turma"], pd.Series)    