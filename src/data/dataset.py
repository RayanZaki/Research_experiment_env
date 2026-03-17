"""Dataset classes for structured data loading."""

import pandas as pd
from torch.utils.data import Dataset
import torch


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular (DataFrame) data.

    Args:
        df: DataFrame containing features and target.
        target_column: Name of the target column.
        feature_columns: Feature column names (default: all except target).
        task: "classification" or "regression" — controls target dtype.
    """

    def __init__(self, df: pd.DataFrame, target_column: str,
                 feature_columns: list[str] | None = None,
                 task: str = "classification"):
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]
        self.X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
        if task == "classification":
            self.y = torch.tensor(df[target_column].values, dtype=torch.long)
        else:
            self.y = torch.tensor(df[target_column].values, dtype=torch.float32)
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    print("dataset module ready")
