"""Data acquisition and transformation utilities.

This module supports:
1) Downloading external sources into data/external (Git repos, files, zip archives)
2) Transforming heterogeneous inputs into data/transformed via adapters
    (csv adapter + generic file/dir copy adapters)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.data.load_data import load_csv
from src.data.preprocess import clean_dataframe, handle_missing, scale_features
from src.features.build_features import add_derived_features


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clone_github_repo(repo_url: str, destination: str, branch: str | None = None, depth: int = 1) -> str:
    """Clone a GitHub repository into destination.

    If destination already exists and is non-empty, the clone is skipped.
    """
    dest = Path(destination)
    if dest.exists() and any(dest.iterdir()):
        print(f"[transform] Skip clone (already exists): {destination}")
        return str(dest)

    _ensure_dir(str(dest.parent))

    cmd = ["git", "clone", "--depth", str(depth)]
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, str(dest)]

    print(f"[transform] Cloning {repo_url} -> {destination}")
    subprocess.run(cmd, check=True)
    return str(dest)


def download_file(url: str, destination: str, timeout: int = 120) -> str:
    """Download a remote file into destination path."""
    _ensure_dir(os.path.dirname(destination) or ".")
    print(f"[transform] Downloading {url} -> {destination}")

    request = urllib.request.Request(url, headers={"User-Agent": "ResearchExperimentEnv/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(destination, "wb") as f:
        shutil.copyfileobj(response, f)
    return destination


def extract_zip(zip_path: str, destination_dir: str) -> str:
    """Extract a zip archive to destination_dir."""
    _ensure_dir(destination_dir)
    print(f"[transform] Extracting {zip_path} -> {destination_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination_dir)
    return destination_dir


def run_external_downloads(external_root: str, cfg: dict) -> None:
    """Run download steps defined in cfg into data/external.

    Expected keys:
      github_repos: [{url, dest?, branch?, depth?}, ...]
      files:        [{url, dest?}, ...]
      zip_files:    [{url, dest?, extract_to?} OR {path, extract_to?}, ...]
    """
    _ensure_dir(external_root)

    for repo in cfg.get("github_repos", []):
        url = repo["url"]
        dest_name = repo.get("dest") or url.rstrip("/").split("/")[-1].replace(".git", "")
        dest = os.path.join(external_root, dest_name)
        clone_github_repo(url, dest, branch=repo.get("branch"), depth=int(repo.get("depth", 1)))

    for item in cfg.get("files", []):
        url = item["url"]
        dest_name = item.get("dest") or url.split("/")[-1]
        dest = os.path.join(external_root, dest_name)
        download_file(url, dest)

    for item in cfg.get("zip_files", []):
        if "url" in item:
            url = item["url"]
            dest_name = item.get("dest") or url.split("/")[-1]
            zip_path = os.path.join(external_root, dest_name)
            download_file(url, zip_path)
        else:
            zip_rel_or_abs = item["path"]
            zip_path = zip_rel_or_abs if os.path.isabs(zip_rel_or_abs) else os.path.join(external_root, zip_rel_or_abs)

        extract_to = item.get("extract_to") or Path(zip_path).stem
        extract_dir = extract_to if os.path.isabs(extract_to) else os.path.join(external_root, extract_to)
        extract_zip(zip_path, extract_dir)


def _root_for_source(source: str, paths_cfg: dict) -> str:
    mapping = {
        "external": paths_cfg.get("data_external", "data/external"),
        "transformed": paths_cfg.get("data_transformed", "data/transformed"),
        "processed": paths_cfg.get("data_processed", "data/processed"),
    }
    return mapping.get(source, mapping["external"])


def _detect_adapter(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        return "copy_dir"
    if p.suffix.lower() == ".csv":
        return "csv"
    return "copy_file"


def _resolve_item_path(item: dict, paths_cfg: dict) -> str:
    path = item["path"]
    if os.path.isabs(path):
        return path

    source = item.get("source", "external")
    return os.path.join(_root_for_source(source, paths_cfg), path)


def _normalize_input_items(data_cfg: dict, explicit_inputs: list[str] | None = None) -> list[dict]:
    if explicit_inputs:
        return [{"path": p} for p in explicit_inputs]

    transform_cfg = data_cfg.get("transform", {})
    configured_items = transform_cfg.get("inputs", [])

    # Backward compatibility: keep support for input_files/source_dirs
    if not configured_items:
        input_files = transform_cfg.get("input_files", [])
        source_dirs = transform_cfg.get("source_dirs", ["external"])
        for f in input_files:
            configured_items.append({"path": f, "source": source_dirs[0] if source_dirs else "external"})

    normalized: list[dict] = []
    for item in configured_items:
        if isinstance(item, str):
            normalized.append({"path": item})
        elif isinstance(item, dict) and "path" in item:
            normalized.append(dict(item))
        else:
            raise ValueError(f"Invalid transform input item: {item}")
    return normalized


def resolve_input_items(paths_cfg: dict, data_cfg: dict, explicit_inputs: list[str] | None = None) -> list[dict]:
    """Resolve transform inputs with adapter metadata.

    Returns a list like:
      {"path": "...", "adapter": "csv|copy_file|copy_dir", "output": "optional/relative/path"}
    """
    normalized = _normalize_input_items(data_cfg=data_cfg, explicit_inputs=explicit_inputs)

    resolved: list[dict] = []
    for item in normalized:
        resolved_path = _resolve_item_path(item, paths_cfg)
        if not os.path.exists(resolved_path):
            print(f"[transform] Warning: input not found: {resolved_path}")
            continue

        adapter = item.get("adapter") or _detect_adapter(resolved_path)
        resolved.append(
            {
                "path": resolved_path,
                "adapter": adapter,
                "output": item.get("output"),
            }
        )

    return resolved


def transform_csv_inputs(
    input_paths: list[str],
    output_path: str,
    preprocessing_cfg: dict,
    target_column: str = "target",
    include_source_column: bool = True,
) -> str:
    """CSV adapter: transform one or multiple CSV files and save unified CSV."""
    if not input_paths:
        raise ValueError("No CSV input files resolved for transformation.")

    transformed_frames: list[pd.DataFrame] = []

    for in_path in input_paths:
        print(f"[transform] Loading CSV: {in_path}")
        df = load_csv(in_path)
        df = clean_dataframe(df, drop_duplicates=preprocessing_cfg.get("drop_duplicates", True))
        df = handle_missing(df, strategy=preprocessing_cfg.get("handle_missing", "drop"))

        scale = preprocessing_cfg.get("scale")
        if scale and scale != "null":
            cols = [c for c in df.columns if c != target_column]
            if cols:
                df, _ = scale_features(df, method=scale, columns=cols)

        df = add_derived_features(df)

        if include_source_column:
            df["_source_file"] = os.path.basename(in_path)

        transformed_frames.append(df)

    merged = pd.concat(transformed_frames, ignore_index=True)
    _ensure_dir(os.path.dirname(output_path) or ".")
    merged.to_csv(output_path, index=False)

    print(f"[transform] Wrote transformed CSV: {output_path} ({len(merged)} rows)")
    return output_path


def transform_items(
    items: list[dict],
    transformed_root: str,
    preprocessing_cfg: dict,
    target_column: str = "target",
    csv_output_file: str = "transformed_dataset.csv",
    include_source_column: bool = True,
) -> dict:
    """Run transform adapters for heterogeneous inputs.

    - csv inputs are merged into one CSV file
    - non-csv files/dirs are copied into data/transformed/assets (or custom output path)
    """
    if not items:
        raise ValueError("No input items resolved for transformation.")

    _ensure_dir(transformed_root)
    assets_root = os.path.join(transformed_root, "assets")
    _ensure_dir(assets_root)

    csv_inputs: list[str] = []
    copied_outputs: list[str] = []

    for item in items:
        src = item["path"]
        adapter = item.get("adapter", "copy_file")
        output_rel = item.get("output")
        dest_base = os.path.join(transformed_root, output_rel) if output_rel else os.path.join(assets_root, os.path.basename(src))

        if adapter == "csv":
            csv_inputs.append(src)
            continue

        if adapter in ("copy_file", "file"):
            _ensure_dir(os.path.dirname(dest_base) or ".")
            shutil.copy2(src, dest_base)
            print(f"[transform] Copied file: {src} -> {dest_base}")
            copied_outputs.append(dest_base)
            continue

        if adapter in ("copy_dir", "dir"):
            if os.path.exists(dest_base):
                shutil.rmtree(dest_base)
            shutil.copytree(src, dest_base)
            print(f"[transform] Copied directory: {src} -> {dest_base}")
            copied_outputs.append(dest_base)
            continue

        raise ValueError(f"Unsupported adapter '{adapter}' for input '{src}'")

    csv_output = None
    if csv_inputs:
        csv_output = os.path.join(transformed_root, csv_output_file)
        transform_csv_inputs(
            input_paths=csv_inputs,
            output_path=csv_output,
            preprocessing_cfg=preprocessing_cfg,
            target_column=target_column,
            include_source_column=include_source_column,
        )

    return {"csv_output": csv_output, "copied_outputs": copied_outputs}
