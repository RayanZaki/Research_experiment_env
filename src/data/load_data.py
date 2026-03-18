"""Functions for loading tabular data files."""

import os
import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_csv(filepath)


def load_all_csvs(directory: str) -> dict[str, pd.DataFrame]:
    """Load all CSV files from a directory into a dict keyed by filename."""
    dataframes = {}
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".csv"):
            dataframes[fname] = pd.read_csv(os.path.join(directory, fname))
    return dataframes


if __name__ == "__main__":
    print("load_data module ready")
