#!/usr/bin/env python3
"""Evaluate a saved PyTorch model on a dataset.

Usage:
    python scripts/evaluate.py --run outputs/experiments/default_20240101_120000 --data data/transformed/test.csv
    python scripts/evaluate.py --no-gpu --run <run> --data <csv>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.utils.gpu import setup_gpu
from src.utils.io import load_json, save_json
from src.utils.reproducibility import get_device
from src.data.load_data import load_csv
from src.data.preprocess import clean_dataframe, handle_missing
from src.data.dataset import TabularDataset
from src.models.architectures import build_model
from src.models.train import load_model
from src.models.evaluate import predict, classification_report, regression_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PyTorch model.")
    parser.add_argument("--run", required=True, help="Path to an experiment run directory")
    parser.add_argument("--data", required=True, help="Path to evaluation data CSV")
    parser.add_argument("--output", help="Path to save evaluation results JSON")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--no-gpu", action="store_true",
                           help="Use all available GPUs without prompting")
    gpu_group.add_argument("--gpus", metavar="IDS",
                           help="Set CUDA_VISIBLE_DEVICES directly (e.g. '0,1')")
    args = parser.parse_args()

    # ── GPU setup ──
    setup_gpu(no_gpu=args.no_gpu, gpus=args.gpus)
    device = get_device()

    # Load run config to reconstruct model
    cfg_path = os.path.join(args.run, "config.yaml")
    cfg = load_json(cfg_path)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    target = data_cfg.get("target_column", "target")

    # Load and reconstruct model
    model_path = os.path.join(args.run, "artifacts", "model.pt")
    print(f"Loading model from {model_path}")
    import torch
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    meta = checkpoint.get("metadata", {})
    model = build_model(meta["model_name"], meta["input_dim"], meta["output_dim"],
                        meta.get("params", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    task = meta.get("task", "classification")

    # Load data
    print(f"Loading data from {args.data}")
    df = load_csv(args.data)
    df = clean_dataframe(df)
    df = handle_missing(df, strategy="drop")

    dataset = TabularDataset(df, target, task=task)
    loader = DataLoader(dataset, batch_size=64, num_workers=0)

    # Evaluate
    y_true, y_pred = predict(model, loader, device, task=task)
    if task == "classification":
        metrics = classification_report(y_true, y_pred)
    else:
        metrics = regression_report(y_true, y_pred)

    print(f"Metrics: {metrics}")

    out_path = args.output or os.path.join(args.run, "eval_results.json")
    save_json(metrics, out_path)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
