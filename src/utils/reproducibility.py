"""Seed and determinism helpers for reproducible experiments."""

import os
import random

import numpy as np


def seed_everything(seed: int = 42):
    """Set seeds for Python, NumPy, and (optionally) PyTorch / TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"[reproducibility] All seeds set to {seed}")


def get_device() -> str:
    """Return the best available device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


if __name__ == "__main__":
    seed_everything()
    print(f"Device: {get_device()}")
