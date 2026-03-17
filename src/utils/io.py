"""I/O helpers: checkpoint saving/loading, config utilities."""

import json
import os

import torch
import yaml


def save_checkpoint(state: dict, filepath: str):
    """Save a training checkpoint to disk using torch.save."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, device: str = "cpu") -> dict:
    """Load a training checkpoint from disk."""
    return torch.load(filepath, map_location=device, weights_only=False)


def load_yaml(filepath: str) -> dict:
    """Load a YAML config file into a dict."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_json(data: dict, filepath: str):
    """Save a dict as a pretty-printed JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> dict:
    """Load a JSON file into a dict."""
    with open(filepath, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print("utils.io module ready")
