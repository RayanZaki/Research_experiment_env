.PHONY: help train evaluate predict transform preprocess test clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Data ──
transform:  ## Download external sources and build data/transformed
	python scripts/transform.py --config configs/default.yaml

download-data:  ## Only download external sources into data/external
	python scripts/transform.py --config configs/default.yaml --download-only

preprocess:  ## Run the preprocessing pipeline
	python scripts/preprocess.py --config configs/default.yaml

# ── Training ──
train:  ## Train a model with default config
	python scripts/train.py --config configs/default.yaml

train-custom:  ## Train with overrides (e.g. make train-custom ARGS="--override model=transformer experiment.seed=99")
	python scripts/train.py --config configs/default.yaml $(ARGS)

# ── Evaluation ──
evaluate:  ## Evaluate latest run (set RUN=path/to/run)
	python scripts/evaluate.py --run $(RUN) --data data/transformed/test.csv

predict:  ## Run inference (set RUN=path/to/run DATA=path/to/csv)
	python scripts/predict.py --run $(RUN) --data $(DATA)

# ── Testing ──
test:  ## Run unit tests
	python -m pytest tests/ -v

# ── Cleanup ──
clean:  ## Remove caches and compiled files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
