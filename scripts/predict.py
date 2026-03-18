#!/usr/bin/env python3
"""Run inference with a trained PyTorch model and save predictions.

Usage:
    python scripts/predict.py --run outputs/experiments/default_20240101_120000 --data data/transformed/new_data.csv
    python scripts/predict.py --no-gpu --run <run> --data <csv>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.gpu import setup_gpu
from src.utils.io import load_json
from src.utils.reproducibility import get_device
from src.data.load_data import load_csv
from src.data.preprocess import clean_dataframe, handle_missing
from src.models.architectures import build_model


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained PyTorch model.")
    parser.add_argument("--run", required=True, help="Path to an experiment run directory")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="outputs/predictions/predictions.csv", help="Output CSV path")
    parser.add_argument("--drop-columns", nargs="*", default=[], help="Columns to drop before prediction")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--no-gpu", action="store_true",
                           help="Use all available GPUs without prompting")
    gpu_group.add_argument("--gpus", metavar="IDS",
                           help="Set CUDA_VISIBLE_DEVICES directly (e.g. '0,1')")
    args = parser.parse_args()

    # ── GPU setup ──
    setup_gpu(no_gpu=args.no_gpu, gpus=args.gpus)
    device = get_device()

    # Load run config and reconstruct model
    model_path = os.path.join(args.run, "artifacts", "model.pt")
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    meta = checkpoint.get("metadata", {})
    model = build_model(meta["model_name"], meta["input_dim"], meta["output_dim"],
                        meta.get("params", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    task = meta.get("task", "classification")

    # Load data
    print(f"Loading data from {args.data}")
    df = load_csv(args.data)
    df = clean_dataframe(df)
    df = handle_missing(df, strategy="drop")

    if args.drop_columns:
        df = df.drop(columns=args.drop_columns, errors="ignore")

    # Run inference
    X = torch.tensor(df.select_dtypes(include="number").values, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(X)
        if task == "classification":
            predictions = output.argmax(dim=1).cpu().numpy()
        else:
            predictions = output.squeeze(-1).cpu().numpy()

    result = df.copy()
    result["prediction"] = predictions

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output} ({len(result)} rows)")


if __name__ == "__main__":
    main()
