"""Feature engineering and selection."""

import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived / engineered features to the DataFrame."""
    # TODO: Add project-specific feature engineering
    return df


def select_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select a subset of columns."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    return df[columns]


if __name__ == "__main__":
    print("build_features module ready")
