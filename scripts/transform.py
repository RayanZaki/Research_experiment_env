#!/usr/bin/env python3
"""Download external data sources and transform heterogeneous data into data/transformed.

Usage:
    python scripts/transform.py --config configs/default.yaml
    python scripts/transform.py --download-only
    python scripts/transform.py --transform-only --inputs data/external/train.csv data/external/metadata
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_yaml
from src.data.transform import run_external_downloads, resolve_input_items, transform_items


def _deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    defaults = cfg.pop("defaults", [])

    for default in defaults:
        if isinstance(default, dict):
            for group, name in default.items():
                sub_path = os.path.join(os.path.dirname(config_path), group, f"{name}.yaml")
                if os.path.exists(sub_path):
                    _deep_update(cfg, load_yaml(sub_path))
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Download and transform datasets.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--download-only", action="store_true", help="Only run external data downloads")
    parser.add_argument("--transform-only", action="store_true", help="Only run transformation step")
    parser.add_argument("--inputs", nargs="*", default=None,
                        help="Input files/directories (absolute or relative); overrides config transform.inputs")
    parser.add_argument("--output", help="Output transformed CSV path (overrides config, CSV adapter only)")
    args = parser.parse_args()

    if args.download_only and args.transform_only:
        raise ValueError("Cannot use both --download-only and --transform-only")

    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})

    external_root = paths_cfg.get("data_external", "data/external")
    transformed_root = paths_cfg.get("data_transformed", "data/transformed")

    external_sources_cfg = cfg.get("external_sources", {})

    if not args.transform_only:
        print("[transform] Running external downloads...")
        run_external_downloads(external_root=external_root, cfg=external_sources_cfg)

    if not args.download_only:
        print("[transform] Running transformation...")
        items = resolve_input_items(paths_cfg=paths_cfg, data_cfg=data_cfg, explicit_inputs=args.inputs)

        transform_cfg = data_cfg.get("transform", {})
        out_name = transform_cfg.get("output_file") or f"transformed_{data_cfg.get('file', 'dataset.csv')}"
        csv_output_file = os.path.basename(args.output) if args.output else out_name

        preproc = data_cfg.get("preprocessing", {})
        target = data_cfg.get("target_column", "target")

        result = transform_items(
            items=items,
            transformed_root=transformed_root,
            preprocessing_cfg=preproc,
            target_column=target,
            csv_output_file=csv_output_file,
            include_source_column=bool(transform_cfg.get("include_source_column", True)),
        )

        if args.output and result.get("csv_output"):
            desired = args.output
            current = result["csv_output"]
            if os.path.abspath(desired) != os.path.abspath(current):
                os.makedirs(os.path.dirname(desired) or ".", exist_ok=True)
                os.replace(current, desired)
                print(f"[transform] Moved CSV output: {current} -> {desired}")

    print("[transform] Done")


if __name__ == "__main__":
    main()
