#!/usr/bin/env python3
"""Train a PyTorch model using a YAML config.

Usage:
    python scripts/train.py                              # prompt for GPUs, use default config
    python scripts/train.py --no-gpu                     # use all available GPUs, no prompt
    python scripts/train.py --gpus 0,1                   # specify GPUs directly
    python scripts/train.py --config configs/default.yaml --override experiment.seed=123
"""

import argparse
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.utils.gpu import setup_gpu
from src.utils.io import load_yaml
from src.utils.reproducibility import seed_everything, get_device
from src.data.load_data import load_csv
from src.data.preprocess import clean_dataframe, handle_missing, scale_features
from src.data.dataset import TabularDataset
from src.features.build_features import add_derived_features
from src.models.architectures import build_model
from src.models.train import train_model, save_model
from src.models.evaluate import predict, classification_report, regression_report
from src.tracking.logger import ExperimentLogger
from src.tracking.wandb_tracker import WandbTracker


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_override(s: str):
    """Parse 'a.b.c=val' into a nested dict."""
    key, _, val = s.partition("=")
    # Try to cast to int/float/bool
    for caster in (int, float):
        try:
            val = caster(val)
            break
        except (ValueError, TypeError):
            continue
    else:
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        elif val.lower() == "null":
            val = None

    parts = key.split(".")
    d = {}
    current = d
    for p in parts[:-1]:
        current[p] = {}
        current = current[p]
    current[parts[-1]] = val
    return d


def _resolve_data_root(paths_cfg: dict, source: str) -> str:
    mapping = {
        "external": paths_cfg.get("data_external", "data/external"),
        "transformed": paths_cfg.get("data_transformed", "data/transformed"),
        "processed": paths_cfg.get("data_processed", "data/processed"),
    }
    if source not in mapping:
        raise ValueError(f"Unknown data source '{source}'. Choose from: {list(mapping)}")
    return mapping[source]


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides: key.subkey=value")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--no-gpu", action="store_true",
                           help="Use all available GPUs without prompting")
    gpu_group.add_argument("--gpus", metavar="IDS",
                           help="Set CUDA_VISIBLE_DEVICES directly (e.g. '0,1')")
    args = parser.parse_args()

    # ── GPU setup (before any CUDA calls) ──
    setup_gpu(no_gpu=args.no_gpu, gpus=args.gpus)

    # ── Load config ──
    cfg = load_yaml(args.config)

    # Merge sub-configs (model, data) referenced by defaults
    defaults = cfg.pop("defaults", [])
    for default in defaults:
        if isinstance(default, dict):
            for group, name in default.items():
                sub_path = os.path.join(os.path.dirname(args.config), group, f"{name}.yaml")
                if os.path.exists(sub_path):
                    sub_cfg = load_yaml(sub_path)
                    _deep_update(cfg, sub_cfg)

    # Apply CLI overrides
    for ov in args.override:
        _deep_update(cfg, _parse_override(ov))

    exp_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})

    # ── Setup ──
    seed_everything(exp_cfg.get("seed", 42))
    device = get_device()
    print(f"Device: {device}")

    logger = ExperimentLogger(
        experiment_name=exp_cfg.get("name", "default"),
        base_dir=paths_cfg.get("experiments", "outputs/experiments"),
    )
    logger.log_config(cfg)

    # ── W&B tracking (optional) ──
    wandb_cfg = cfg.get("wandb", {})
    wb = None
    if wandb_cfg.get("enabled", False):
        wb = WandbTracker(
            project=wandb_cfg.get("project", cfg.get("project", {}).get("name", "default")),
            experiment_name=exp_cfg.get("name", "default"),
            config=cfg,
            tags=wandb_cfg.get("tags", []),
            notes=wandb_cfg.get("notes", ""),
            mode=wandb_cfg.get("mode", "online"),
        )
        print("[wandb] Run initialized")

    # ── Pipeline ──
    # 1. Load data
    data_source = data_cfg.get("source", "transformed")
    data_root = _resolve_data_root(paths_cfg, data_source)
    data_file = os.path.join(data_root, data_cfg.get("file", "dataset.csv"))

    print(f"Loading data from {data_file} ...")
    df = load_csv(data_file)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Preprocess
    preproc = data_cfg.get("preprocessing", {})
    df = clean_dataframe(df, drop_duplicates=preproc.get("drop_duplicates", True))
    df = handle_missing(df, strategy=preproc.get("handle_missing", "drop"))
    target = data_cfg.get("target_column", "target")
    if preproc.get("scale") and preproc["scale"] != "null":
        feature_cols = [c for c in df.columns if c != target]
        df, _scaler = scale_features(df, method=preproc["scale"], columns=feature_cols)

    # 3. Feature engineering
    df = add_derived_features(df)

    # 4. Determine task type
    y = df[target]
    if y.dtype.kind == "f" and y.nunique() > 20:
        task = "regression"
        output_dim = 1
    else:
        task = "classification"
        output_dim = int(y.nunique())

    # 5. Split into train / val / test
    from sklearn.model_selection import train_test_split
    test_size = data_cfg.get("test_size", exp_cfg.get("test_size", 0.2))
    val_size = data_cfg.get("val_size", 0.1)
    stratify = y if data_cfg.get("split_strategy") == "stratified" else None
    seed = exp_cfg.get("seed", 42)

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify,
    )
    stratify_val = df_train[target] if stratify is not None else None
    df_train, df_val = train_test_split(
        df_train, test_size=val_size / (1 - test_size), random_state=seed,
        stratify=stratify_val,
    )
    print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # 6. Create DataLoaders
    model_params = model_cfg.get("params", {})
    batch_size = model_params.get("batch_size", 64)
    num_workers = exp_cfg.get("num_workers", 4)

    train_ds = TabularDataset(df_train, target, task=task)
    val_ds = TabularDataset(df_val, target, task=task)
    test_ds = TabularDataset(df_test, target, task=task)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    # 7. Build model
    model_name = model_cfg.get("name", "mlp")
    input_dim = train_ds.X.shape[1]
    model = build_model(model_name, input_dim, output_dim, model_params)
    print(f"  Model: {model_name} ({task}, in={input_dim}, out={output_dim})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 8. Train
    model, history = train_model(
        model, train_loader, val_loader, device=device,
        epochs=model_params.get("epochs", 50),
        lr=model_params.get("lr", 0.001),
        weight_decay=model_params.get("weight_decay", 0.0001),
        patience=model_params.get("patience", 10),
        scheduler_type=model_params.get("scheduler", "plateau"),
        task=task, logger=logger, wandb_tracker=wb,
    )

    # 9. Evaluate on test set
    y_true, y_pred = predict(model, test_loader, device, task=task)
    if task == "classification":
        metrics = classification_report(y_true, y_pred)
    else:
        metrics = regression_report(y_true, y_pred)
    logger.log_metrics(metrics)
    if wb is not None:
        wb.log_metrics(metrics)
    print(f"  Test metrics: {metrics}")

    # 10. Save model
    model_path = os.path.join(logger.run_dir, "artifacts", "model.pt")
    metadata = {"model_name": model_name, "input_dim": input_dim,
                "output_dim": output_dim, "task": task, "params": model_params}
    save_model(model, model_path, metadata=metadata)

    # Save training history plot
    from src.visualization.plot import plot_training_history
    plot_training_history(history, output_dir=os.path.join(logger.run_dir, "artifacts"))

    # Upload model artifact to W&B
    if wb is not None:
        wb.log_artifact(model_path, artifact_type="model")
        wb.finish()

    print(f"\nExperiment complete: {logger.run_dir}")
    print(f"  Summary: {logger.summary()}")


if __name__ == "__main__":
    main()
