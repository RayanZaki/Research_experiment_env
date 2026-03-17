"""PyTorch model evaluation utilities."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str,
            task: str = "classification") -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        if task == "classification":
            preds = output.argmax(dim=1).cpu().numpy()
        else:
            preds = output.squeeze(-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())
    return np.concatenate(all_targets), np.concatenate(all_preds)


def classification_report(y_true, y_pred) -> dict:
    """Return a dict of common classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def regression_report(y_true, y_pred) -> dict:
    """Return a dict of common regression metrics."""
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


if __name__ == "__main__":
    print("evaluate module ready")
