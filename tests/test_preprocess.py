"""Tests for data preprocessing."""

import pandas as pd
from src.data.preprocess import clean_dataframe, handle_missing


def test_clean_dataframe_drops_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    result = clean_dataframe(df)
    assert len(result) == 2


def test_handle_missing_drop():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    result = handle_missing(df, strategy="drop")
    assert result.isna().sum().sum() == 0


def test_handle_missing_zero():
    df = pd.DataFrame({"a": [1, None, 3]})
    result = handle_missing(df, strategy="zero")
    assert result["a"].iloc[1] == 0
