# Copilot Instructions for Research_experiment_env

## Mission
This file is the persistent architecture contract for this repository. Always follow it when making changes.

## Framework architecture (source of truth)

### 1) Entry points
- `scripts/train.py`: end-to-end training pipeline (config load/merge, data load, preprocess, feature build, split, train, evaluate, save artifacts).
- `scripts/preprocess.py`: preprocessing-only pipeline, writes processed CSV.
- `scripts/evaluate.py`: evaluate a saved run/model on labeled data.
- `scripts/predict.py`: run inference on unlabeled/new data.

### 2) Config system
- Base config: `configs/default.yaml`.
- Sub-config groups:
  - `configs/data/*.yaml`
  - `configs/model/*.yaml`
- Merge order:
  1. base config
  2. files in `defaults`
  3. CLI overrides
- Path keys in `paths` define canonical locations (`data_raw`, `data_processed`, `data_external`, `outputs/*`, `logs`).

### 3) Data lifecycle
- `data/raw`: primary immutable input data.
- `data/external`: third-party or auxiliary enrichment data.
- `data/processed`: outputs of cleaning/feature preparation.

### 4) Core library layout
- `src/data`: loading, cleaning, dataset objects.
- `src/features`: feature engineering.
- `src/models`: architectures, training, evaluation helpers.
- `src/tracking`: run logging and optional Weights & Biases integration.
- `src/utils`: I/O, reproducibility, GPU helpers.
- `src/visualization`: plots and charts.

### 5) Outputs and reproducibility
- `outputs/experiments/<run_timestamp>` stores config snapshot, metrics, and artifacts.
- `outputs/predictions` stores inference files.
- `logs` stores runtime logs.
- `tests` validates behavior.

## Update policy (mandatory)
When anything in repository architecture, defaults, paths, scripts, module layout, or workflow changes:
1. Update this file immediately.
2. Update `README.md` if user-facing behavior changed.
3. **Update the `create_project` script/templates in the same change** so new projects scaffold the latest architecture.

## Consistency checklist for every structural change
- [ ] `configs/` examples still match script behavior.
- [ ] `scripts/` CLI examples still valid.
- [ ] `src/` module references and imports still correct.
- [ ] `README.md` architecture section updated.
- [ ] `create_project` templates updated to mirror current repository.
- [ ] Tests updated/added if behavior changed.
