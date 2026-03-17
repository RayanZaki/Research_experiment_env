"""PyTorch model training utilities."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model, train_loader: DataLoader, val_loader: DataLoader | None,
                device: str, epochs: int = 50, lr: float = 0.001,
                weight_decay: float = 0.0001, patience: int = 10,
                scheduler_type: str = "plateau", task: str = "classification",
                logger=None, wandb_tracker=None):
    """Train a PyTorch model with early stopping.

    Returns (model, history) where history is a dict of metric lists.
    """
    model = model.to(device)

    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2)
    else:
        scheduler = None

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss, n = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
            n += len(X_batch)
        train_loss = total_loss / n
        history["train_loss"].append(train_loss)

        # ── Validate ──
        val_loss = None
        if val_loader is not None:
            model.eval()
            total_loss, n = 0.0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    total_loss += loss.item() * len(X_batch)
                    n += len(X_batch)
            val_loss = total_loss / n
            history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            # Scheduler step
            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        else:
            if scheduler is not None and scheduler_type != "plateau":
                scheduler.step()

        if logger:
            logger.log_metric("train_loss", train_loss, step=epoch)
            if val_loss is not None:
                logger.log_metric("val_loss", val_loss, step=epoch)

        if wandb_tracker:
            epoch_metrics = {"train_loss": train_loss, "epoch": epoch}
            if val_loss is not None:
                epoch_metrics["val_loss"] = val_loss
            wandb_tracker.log_metrics(epoch_metrics, step=epoch)

        if epoch % 10 == 0 or epoch == 1:
            msg = f"  Epoch {epoch:>4d} | train_loss: {train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val_loss: {val_loss:.4f}"
            print(msg)

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def save_model(model, filepath: str, metadata: dict | None = None):
    """Save a PyTorch model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {"model_state_dict": model.state_dict()}
    if metadata:
        state["metadata"] = metadata
    torch.save(state, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath: str, device: str = "cpu"):
    """Load weights into a PyTorch model."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint.get("metadata", {})


if __name__ == "__main__":
    print("train module ready")
