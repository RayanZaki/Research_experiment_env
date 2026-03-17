"""Functions for cleaning and transforming data."""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, reset index."""
    if drop_duplicates:
        df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Handle missing values with the given strategy."""
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    elif strategy == "zero":
        return df.fillna(0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def scale_features(df: pd.DataFrame, method: str = "standard", columns: list[str] | None = None):
    """Scale numeric columns. Returns (scaled_df, scaler)."""
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(f"Unknown scaling method: {method}. Choose from {list(scalers)}")

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    scaler = scalers[method]()
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


if __name__ == "__main__":
    print("preprocess module ready")
