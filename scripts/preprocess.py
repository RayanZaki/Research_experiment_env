#!/usr/bin/env python3
"""Run the data preprocessing pipeline only.

Usage:
    python scripts/preprocess.py --config configs/default.yaml
    python scripts/preprocess.py --input data/raw/dataset.csv --output data/processed/clean.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_yaml
from src.data.load_data import load_csv
from src.data.preprocess import clean_dataframe, handle_missing, scale_features
from src.features.build_features import add_derived_features


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw data.")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--input", help="Path to input CSV (overrides config)")
    parser.add_argument("--output", help="Path to save processed CSV (overrides config)")
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml(args.config)
        # Merge sub-configs
        defaults = cfg.pop("defaults", [])
        for default in defaults:
            if isinstance(default, dict):
                for group, name in default.items():
                    sub_path = os.path.join(os.path.dirname(args.config), group, f"{name}.yaml")
                    if os.path.exists(sub_path):
                        sub_cfg = load_yaml(sub_path)
                        cfg.update(sub_cfg)
        data_cfg = cfg.get("data", {})
        paths_cfg = cfg.get("paths", {})
    else:
        data_cfg = {}
        paths_cfg = {}

    input_path = args.input or os.path.join(
        paths_cfg.get("data_raw", "data/raw"),
        data_cfg.get("file", "dataset.csv"),
    )
    output_path = args.output or os.path.join(
        paths_cfg.get("data_processed", "data/processed"),
        "processed_" + os.path.basename(input_path),
    )

    print(f"Loading: {input_path}")
    df = load_csv(input_path)
    print(f"  Raw shape: {df.shape}")

    preproc = data_cfg.get("preprocessing", {})
    df = clean_dataframe(df, drop_duplicates=preproc.get("drop_duplicates", True))
    df = handle_missing(df, strategy=preproc.get("handle_missing", "drop"))

    if preproc.get("scale") and preproc["scale"] != "null":
        target = data_cfg.get("target_column", "target")
        feature_cols = [c for c in df.columns if c != target]
        df, _ = scale_features(df, method=preproc["scale"], columns=feature_cols)

    df = add_derived_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Processed shape: {df.shape}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
